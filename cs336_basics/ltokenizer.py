from typing import Iterator, Iterable
from dataclasses import dataclass
import os
import regex as re
from multiprocessing import Pool
import heapq
import json
import argparse
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

num_processes = 16

def __pre_tokenization(args) -> dict[bytes, int]:
    input_path, start, end, special_tokens = args
    chunks = []

    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # 1. decompose the chunk according to special tokens
    spec_indexes_begin = []
    spec_indexes_end = []
    for special_tok in special_tokens:
        for i in range(len(chunk) - len(special_tok) + 1):
            if chunk[i: i+len(special_tok)] == special_tok:
                spec_indexes_begin.append(i)
                spec_indexes_end.append(i+len(special_tok))
    for i in range(len(spec_indexes_begin)):
        prev = spec_indexes_end[i-1] if i > 0 else 0
        chunks.append(chunk[prev: spec_indexes_begin[i]])
    if len(spec_indexes_end) > 0:
        if spec_indexes_end[-1] < len(chunk): 
            chunks.append(chunk[spec_indexes_end[-1]:])
    else:
        chunks.append(chunk)

    # 2. gen pre-tokenization dict
    pre_toks = dict()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            nowstr = m.group()
            nowstr = nowstr.encode('utf-8')
            pre_toks[nowstr] = pre_toks.get(nowstr, 0) + 1
    
    return pre_toks

@dataclass
class TokenPair:
    pairs: list[bytes] # assert len == 2
    frequency: int
    appearing: list[int]
    updated: bool = True
    realtime_frequency: int | None = None

    def __lt__(self, other):
        return (self.frequency > other.frequency) or ((self.frequency == other.frequency) and (self.pairs[0] > other.pairs[0])) or ((self.frequency == other.frequency) and (self.pairs[0] == other.pairs[0]) and (self.pairs[1] > other.pairs[1]))


def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # pretokenization
    pre_toks = dict()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))
        starts, ends = [], []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            starts.append(start)
            ends.append(end)
        
        print('ckpt1')
        
    with Pool(num_processes) as p:
        results = p.map(__pre_tokenization, [(input_path, starts[i], ends[i], special_tokens) for i in range(len(starts))])
    
    print('ckpt2')

    pre_toks = {}
    for res in results:
        pre_toks = pre_toks | res
    for key in pre_toks:
        pre_toks[key] = sum([result.get(key, 0) for result in results])

    # pretokenization done
    print("Pretokenization done")
    # print(len(pre_toks))
    pre_toks = [[key, pre_toks[key], []] for key in pre_toks] # transform to [doc, freq, list[int]] list

    # init byte pairs
    init_bpes = dict()
    init_bytes = set()
    for i, item in tqdm(enumerate(pre_toks), total=len(pre_toks)):
        doc, freq, _ = item
        for j in range(len(doc) - 1):
            s, t = doc[j: j+1], doc[j+1: j+2]
            orig_v = init_bpes.get((s,t), [0, []])
            orig_v[0] += freq
            orig_v[1].append(i)
            if len(orig_v[1]) == 1:
                init_bpes[(s,t)] = orig_v
    heap = []
    bpe_dict = dict() # (tok1, tok2) -> TokenPair
    for k in init_bpes:
        pair = TokenPair([k[0], k[1]], init_bpes[k][0], list(set(init_bpes[k][1])))
        heapq.heappush(heap, pair)
        bpe_dict[(k[0], k[1])] = pair
    
    # initial bytes as tokens
    vocab = dict()
    vocab_reversed = dict()
    init_bytes = [bytes([i]) for i in range(256)]
    for item in init_bytes:
        vocab[len(vocab)] = item
        vocab_reversed[item] = len(vocab) - 1
    for item in tqdm(pre_toks):
        item[2] = [int(ii) for ii in item[0]] # init the tokenized doc
    merges = list()
    
    # main merge procedure
    with tqdm(total=vocab_size - len(special_tokens) - len(vocab), unit='tokens', desc='merging tokens') as pbar:
        while len(vocab) < vocab_size - len(special_tokens):
            pbar.update(1)
            if len(heap) == 0:
                break

            while heap:
                now_token_pair = heapq.heappop(heap)
                if not now_token_pair.updated: # frequency not updated in heap's order
                    # print('Obsolute', now_token_pair)
                    newtoken_pair = TokenPair(now_token_pair.pairs, now_token_pair.realtime_frequency, now_token_pair.appearing)
                    heapq.heappush(heap, newtoken_pair)
                    bpe_dict[(newtoken_pair.pairs[0], newtoken_pair.pairs[1])] = newtoken_pair
                else:
                    break

            if now_token_pair.frequency == 0:
                # nothing to merge
                break
            pre_tok_a, pre_tok_b = now_token_pair.pairs[0], now_token_pair.pairs[1]
            merges.append((pre_tok_a, pre_tok_b))
            new_token = pre_tok_a + pre_tok_b
            vocab[len(vocab)] = new_token
            vocab_reversed[new_token] = len(vocab) - 1

            # print('Merging ', now_token_pair)

            new_bpes = dict() # tuple(bpe1, bpe2) -> [frequency, appearing_list]

            doc_list = now_token_pair.appearing
            for doc_index in doc_list:
                old_toklist = pre_toks[doc_index][2]
                new_toklist = []
                
                # update docs
                i = 0
                find = False
                while i < len(old_toklist):
                    if i+1 < len(old_toklist) and old_toklist[i] == vocab_reversed[pre_tok_a] and old_toklist[i+1] == vocab_reversed[pre_tok_b]:
                        new_toklist.append(len(vocab) - 1)
                        find = True
                        i += 2
                    else:
                        new_toklist.append(old_toklist[i])
                        i += 1
                pre_toks[doc_index][2] = new_toklist

                if find:
                    for i, tok in enumerate(new_toklist):
                        if tok == len(vocab) - 1: # new token added here
                            for j in [0,1]: # add new bpes, before and after
                                if j == 0:
                                    if i == 0: continue 
                                    # new_bpe_a = vocab[new_toklist[i-1]] if new_toklist[i-1] != tok else pre_tok_b
                                    new_bpe_a = vocab[new_toklist[i-1]]
                                    new_bpe_b = new_token
                                if j == 1:
                                    if i == len(new_toklist) - 1: continue
                                    new_bpe_a = new_token
                                    new_bpe_b = vocab[new_toklist[i+1]]
                                    # new_bpe_b = vocab[new_toklist[i+1]] if new_toklist[i+1] != tok else pre_tok_a
                                old_freq, old_appearing = new_bpes.get((new_bpe_a, new_bpe_b), [0, []])
                                old_appearing.append(doc_index)
                                new_bpes[(new_bpe_a, new_bpe_b)] = [old_freq + pre_toks[doc_index][1], old_appearing]
                            
                            for j in [0,1]: # subtract freq from old token pairs
                                if j == 0:
                                    if i == 0: continue
                                    old_bpe_a = vocab[new_toklist[i-1]] if new_toklist[i-1] != tok else pre_tok_b
                                    old_bpe_b = pre_tok_a
                                if j == 1:
                                    if i == len(new_toklist) - 1: continue
                                    old_bpe_a = pre_tok_b
                                    old_bpe_b = vocab[new_toklist[i+1]] if new_toklist[i+1] != tok else pre_tok_a
                                if bpe_dict[(old_bpe_a, old_bpe_b)].realtime_frequency is None:
                                    bpe_dict[(old_bpe_a, old_bpe_b)].realtime_frequency = bpe_dict[(old_bpe_a, old_bpe_b)].frequency - pre_toks[doc_index][1]
                                    bpe_dict[(old_bpe_a, old_bpe_b)].updated = False
                                else:
                                    bpe_dict[(old_bpe_a, old_bpe_b)].realtime_frequency -= pre_toks[doc_index][1]
                                    bpe_dict[(old_bpe_a, old_bpe_b)].updated = False
                                # if old_bpe_a == b't' and old_bpe_b == b'h':
                                    # print('Updated:', bpe_dict[(old_bpe_a, old_bpe_b)])

            # add collected new BPEs into the heap
            # print('Merging ', pre_tok_a, ' and ', pre_tok_b, ' that appear freq =', now_token_pair.frequency, ' and appears in ', now_token_pair.appearing,  ' generates ', new_bpes)

            for k in new_bpes:
                v = new_bpes[k]
                new_token_pair = TokenPair([k[0], k[1]], v[0], list(set(v[1])))
                bpe_dict[k] = new_token_pair
                heapq.heappush(heap, new_token_pair)
                # print('New: ', new_token_pair)

    # add special tokens
    for spec_tokens in special_tokens:
        vocab[len(vocab)] = spec_tokens.encode('utf-8')

    # print(vocab, merges)
    print('tokenizer training done')

    return vocab, merges

def dump(vocab, merges, vocab_filepath, merges_filepath):
    with open(vocab_filepath, 'wb') as f:
        f.writelines([str(k).encode('utf-8') + '<|endoftext|>'.encode('utf-8') + vocab[k] + '\n'.encode('utf-8') for k in vocab])
    with open(merges_filepath, 'wb') as f:
        f.writelines([m[0] + '<|endoftext|>'.encode('utf-8') + m[1] + '\n'.encode('utf-8') for m in merges])

parser = argparse.ArgumentParser()
parser.add_argument('traintext', type=str)
parser.add_argument('vocab_size', type=int)
parser.add_argument('vocab_output_path', type=str, help='a txt file')
parser.add_argument('merges_output_path', type=str, help='a txt file')
if __name__ == '__main__':
    args = parser.parse_args()
    # train_bpe('data/TinyStoriesV2-GPT4-train.txt', 3000, ['<|endoftext|>'])
    # train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 3000, ['<|endoftext|>'])
    # train_bpe('data/TinyStoriesV2-GPT4-tiny.txt', 300, ['<|endoftext|>'])
    # train_bpe('data/TinyStoriesV2-GPT4-tinytiny.txt', 300, ['<|endoftext|>'])
    # vocab, merges = train_bpe(
    #     input_path='tests/fixtures/corpus.en',
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # vocab, merges = train_bpe(
    #     'data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>']
    # )

    # vocab, merges = train_bpe(
    #     'data/owt_train.txt', 32000, ['<|endoftext|>']
    # )
    # print(len(vocab), len(merges))
    # vocab_filepath = 'data/owt_vocab.json'
    # merges_filepath = 'data/owt_merges.txt'
    # dump(vocab, merges, vocab_filepath, merges_filepath)

    vocab, merges = train_bpe(
        args.traintext, args.vocab_size, ['<|endoftext|>']
    )
    print(len(vocab), len(merges))
    dump(vocab, merges, args.vocab_output_path, args.merges_output_path)


class LTokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        pass
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass