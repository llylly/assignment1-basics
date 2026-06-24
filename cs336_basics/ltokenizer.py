from typing import Iterator, Iterable
from dataclasses import dataclass
import os
import regex as re
from multiprocessing import Pool
import heapq
import json
import argparse
import numpy as np
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
    with open(vocab_filepath, 'w') as f:
        for k in vocab:
            print((k, [s for s in vocab[k]]), file=f)
    with open(merges_filepath, 'w') as f:
        for m in merges:
            print(([x for x in m[0]], [x for x in m[1]]), file=f)

class LTokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.int_merges = [(self.inv_vocab[ka], self.inv_vocab[kb]) for ka, kb in self.merges]
        self.int_merges_map = {item: i  for i, item in enumerate(self.int_merges)}

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens is not None:
            special_tok_pattern = r"|".join([re.escape(spec_tokens) for spec_tokens in self.special_tokens])
            special_tok_pattern += r"|"
            self.PAT = special_tok_pattern + self.PAT
            # append to vocab if not there
            for spec_tok in self.special_tokens:
                if spec_tok.encode('utf-8') not in self.inv_vocab:
                    self.vocab[len(self.vocab)] = spec_tok.encode('utf-8')
                    self.inv_vocab[spec_tok.encode('utf-8')] = len(self.vocab) - 1
        print('pattern:', self.PAT)

        self.cache = dict()
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = dict()
        merges = list()
        with open(vocab_filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = eval(line)
                vocab[line[0]] = bytes(line[1])
        with open(merges_filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = eval(line)
                merges.append((bytes(line[0]), bytes(line[1])))
        return LTokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = []
        st = 0
        if self.special_tokens:
            with tqdm(desc='strip special tokens', total=len(text)) as pbar:
                while True:
                    min_p, min_tok = len(text) + 1, None
                    for spec_tok in self.special_tokens:
                        p = text.find(spec_tok, st)
                        if p != -1:
                            if p < min_p:
                                min_p, min_tok = p, spec_tok
                            elif p == min_p and spec_tok > min_tok:
                                min_tok = spec_tok
                    if min_p == len(text) + 1:
                        break
                    if min_p > st: chunks.append(text[st: min_p])
                    chunks.append(min_tok)
                    pbar.update(min_p + len(min_tok) - st)
                    st = min_p + len(min_tok)
        if st < len(text):
            chunks.append(text[st:])

        # encode and cache in the mean time
        ans = []

        with tqdm(unit=' unique words', desc='tokenizing') as pbar:
            with tqdm(unit=' total words', desc='tokenizing') as ppbar:
                with tqdm(unit=' total tokens', desc='tokenizing') as pppbar:
                    for textt in chunks:
                        if self.special_tokens and any([textt == spec_tok for spec_tok in self.special_tokens]):
                            ans.append(self.inv_vocab[textt.encode('utf-8')])
                            pppbar.update(1)
                            ppbar.update(1)
                            continue
                        for m in re.finditer(self.PAT, textt):
                            # print(m.group())
                            if m.group() not in self.cache:
                                pbar.update(1)
                                l = self._real_encode(m.group())
                            else:
                                l = self.cache[m.group()]
                            
                            ans.extend(l)
                            # print(m.group(), '->', l)
                            pppbar.update(len(l))
                            ppbar.update(1)
        
        bytelen = len(text.encode('utf-8'))
        print('total bytes:', bytelen, 'total tokens:', pppbar.n)
        if pppbar.n > 0: print('compression ratio:', bytelen / pppbar.n)
        return ans

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for num in self.encode(text):
                yield num

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')

    def _real_encode(self, w: str):
        wb = w.encode('utf-8')
        l = [self.inv_vocab[wb[i: i+1]] for i in range(len(wb))]

        while True:
            minmerge_id = len(self.int_merges) + 1
            for i in range(len(l) - 1):
                minmerge_id = min(minmerge_id, self.int_merges_map.get((l[i], l[i+1]), minmerge_id))
            if minmerge_id > len(self.int_merges):
                break
            ka, kb = self.int_merges[minmerge_id]
            p = []
            for i in range(len(l) - 1):
                if l[i] == ka and l[i+1] == kb and ((not p) or p[-1] != i-1):
                    p.append(i)
            for j in p[::-1]:
                del l[j+1]
                del l[j]
                l.insert(j, self.inv_vocab[self.vocab[ka]+self.vocab[kb]])

        # appear_token = set(l)
        # for ka, kb in self.int_merges:
        #     if ka in appear_token and kb in appear_token:
        #         p = []
        #         newtok = self.inv_vocab[self.vocab[ka]+self.vocab[kb]]
        #         for i in range(len(l) - 1):
        #             if l[i] == ka and l[i+1] == kb and ((not p) or p[-1] != i-1):
        #                 p.append(i)
        #         # print(ka, kb, l, p)
        #         for j in p[::-1]:
        #             del l[j + 1]
        #             del l[j]
        #             l.insert(j, newtok)
        #         appear_token = set(l)
        self.cache[w] = l
        return l



parser = argparse.ArgumentParser()
parser.add_argument('vocab_output_path', type=str, help='a txt file')
parser.add_argument('merges_output_path', type=str, help='a txt file')
parser.add_argument('--text_path', type=str, help='a txt file to encode')
parser.add_argument('--tokenize_output_path', type=str, help='a numpy .npy file for tokenized output in numpy.array of uint16')
parser.add_argument('--traintext', type=str)
parser.add_argument('--vocab_size', type=int)
"""
Training usage:
    uv run python cs336_basics/ltokenizer.py data/owt_vocab.txt data/owt_merges.txt --traintext data/owt_train.txt --vocab_size 32000 
    uv run python cs336_basics/ltokenizer.py data/TinyStoriesV2-GPT4-vocab.txt data/TinyStoriesV2-GPT4-merges.txt --traintext data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 
    uv run python cs336_basics/ltokenizer.py data/TinyStoriesV2-GPT4-vocab.txt data/TinyStoriesV2-GPT4-merges.txt --traintext data/TinyStoriesV2-GPT4-valid.txt --vocab_size 10000 
Test usage:
    uv run python cs336_basics/ltokenizer.py data/TinyStoriesV2-GPT4-vocab.txt data/TinyStoriesV2-GPT4-merges.txt --text_path data/TinyStoriesV2-GPT4-tinytiny.txt --tokenize_output_path data/TinyStoriesV2-GPT4-tinytiny.tokenized.npy
    uv run python cs336_basics/ltokenizer.py data/TinyStoriesV2-GPT4-vocab.txt data/TinyStoriesV2-GPT4-merges.txt --text_path data/TinyStoriesV2-GPT4-tiny.txt --tokenize_output_path data/TinyStoriesV2-GPT4-tiny.tokenized.npy
    uv run python cs336_basics/ltokenizer.py data/TinyStoriesV2-GPT4-vocab.txt data/TinyStoriesV2-GPT4-merges.txt --text_path data/TinyStoriesV2-GPT4-valid.txt --tokenize_output_path data/TinyStoriesV2-GPT4-valid.tokenized.npy
    uv run python cs336_basics/ltokenizer.py data/owt_vocab.txt data/owt_merges.txt --text_path data/owt_valid.txt --tokenize_output_path data/owt_valid.tokenized.npy
    uv run python cs336_basics/ltokenizer.py data/TinyStoriesV2-GPT4-vocab.txt data/TinyStoriesV2-GPT4-merges.txt --text_path data/TinyStoriesV2-GPT4-train.txt --tokenize_output_path data/TinyStoriesV2-GPT4-train.tokenized.npy
    uv run python cs336_basics/ltokenizer.py data/owt_vocab.txt data/owt_merges.txt --text_path data/owt_train.txt --tokenize_output_path data/owt_train.tokenized.npy
"""
if __name__ == '__main__':
    args = parser.parse_args()

    if args.traintext is not None and args.vocab_size is not None:
        # training mode
        vocab, merges = train_bpe(
            args.traintext, args.vocab_size, ['<|endoftext|>']
        )
        print(len(vocab), len(merges))
        dump(vocab, merges, args.vocab_output_path, args.merges_output_path)
    else:
        # inference mode
        tokenizer = LTokenizer.from_files(args.vocab_output_path, args.merges_output_path, ['<|endoftext|>'])
        assert args.tokenize_output_path is not None and args.tokenize_output_path.endswith('.npy'), 'Need to store to a npy file'

        splits = 5

        starts, ends = [], []
        with open(args.text_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, splits, '<|endoftext|>'.encode('utf-8'))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            starts.append(start)
            ends.append(end)
        
        # serialize so RAM can afford the cost
        results = []
        for s, e in zip(starts, ends):
            with open(args.text_path, 'rb') as f:
                f.seek(s)
                txt = f.read(e-s).decode('utf-8')
            now_res = np.array(tokenizer.encode(txt), dtype=np.uint16)
            results.append(now_res)
        results = np.concat(results)
        print('Total tokens:', len(results))
        np.save(args.tokenize_output_path, results)
