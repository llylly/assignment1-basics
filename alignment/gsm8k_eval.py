from dataclasses import dataclass, field, asdict
from typing import Literal
import os
import json
import tqdm
import math
import time
from datetime import datetime

import tyro
import wandb
from alignment import vllm_utils
import basics.lmodeling_olmo as lmodeling_olmo
from basics.linference import generate
from alignment import drgrpo_grader

@dataclass
class EvalConfig:
    backend: Literal['native', 'vllm'] = 'native' # 'vllm' / 'native'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16' # or 'float32'
    prompt_type: Literal['r1_zero', 'question_only', 'r1_zero_three_shot_gsm8k'] = 'r1_zero' # or 'question_only'
    max_new_tokens: int = 512
    temperature: float = 1.0
    n: int = 5
    batch_size: int = 5
    first_n_samp: int | None = None
    model_dir: str = 'models/OLMo-2-0425-1B'
    data_dir: str = 'data/gsm8k'
    run_suffix: str = 'baseeval'
    save_dir: str = 'eval'

"""
Example commands:
uv run alignment/gsm8k_eval.py --backend vllm --prompt_type r1_zero
uv run alignment/gsm8k_eval.py --backend vllm --prompt_type r1_zero_three_shot_gsm8k
uv run alignment/gsm8k_eval.py --backend vllm --prompt_type question_only
uv run alignment/gsm8k_eval.py --backend native --prompt_type r1_zero
uv run alignment/gsm8k_eval.py --backend native --prompt_type r1_zero_three_shot_gsm8k
uv run alignment/gsm8k_eval.py --backend native --prompt_type question_only
"""
if __name__ == '__main__':
    config = tyro.cli(EvalConfig)
    nowtime = datetime.now().strftime('_%Y%m%d_%H%M%S')
    run_name = f'gsm8k_test_{config.run_suffix}_{config.backend}_prompt_{config.prompt_type}_temp_{config.temperature}_n_{config.n}_max_new_tokens_{config.max_new_tokens}_{config.model_dir.replace("/", "-")}_{config.dtype}_bs_{config.batch_size}_{nowtime}'

    # wandb
    wandb.init(project='LLLM_eval_gsm8k_test', name=run_name, config=asdict(config), dir=os.path.join(config.save_dir, 'wandb_logs'))
    stime = time.time()

    # load data
    with open(os.path.join(config.data_dir, 'test.jsonl'), 'r') as f:
        test_data = [json.loads(item) for item in f.readlines()]
    
    print('Loading prompt...')
    with open(os.path.join('alignment/prompts', config.prompt_type + '.prompt'), 'r') as f:
        prompt_template = f.read()

    print('Loading model...')
    if config.backend == 'vllm':
        serv_proc = vllm_utils.start_server(config.model_dir, '127.0.0.1', 8080, config.dtype, 0, 42, "auto", 'INFO')
        vllm_utils.wait_for_server('http://127.0.0.1:8080', serv_proc, 60)
    elif config.backend == 'native':
        model, tokenizer = lmodeling_olmo.from_pretrained(config.model_dir, config.dtype)
    
    all_finished = []

    now_cnt = 0
    
    avg_pass1 = {}
    avg_passn = {}

    for item in tqdm.tqdm(test_data):
        raw_question = item['question']
        final_answer = item['answer'][item['answer'].find('####') + 4:].strip()
        templated_question = prompt_template.format(question=raw_question)
        continuations = []
        while len(continuations) < config.n:
            amounts_to_gen = min(config.n - len(continuations), config.batch_size)
            if config.backend == 'vllm':
                ret = vllm_utils.generate_completions('http://127.0.0.1:8080', 'models/OLMo-2-0425-1B', [templated_question], {
                    'temperature': config.temperature,
                    'max_tokens': config.max_new_tokens,
                    'n': config.n,
                    'seed': 42,
                    'stop': ['</answer>'],
                    'include_stop_str_in_output': True,
                }, None)
            elif config.backend == 'native':
                ret = generate(model, [templated_question for _ in range(config.n)], tokenizer, config.max_new_tokens, config.temperature, extra_stop_tokens=['</answer>'], include_stop_str_in_output=True, verbose=False)
            else:
                raise NotImplementedError
            continuations.extend(ret)
        
        finished = {
            'question': item['question'],
            'answer': item['answer'],
            'final_answer': final_answer,
            'continuations': continuations
        }
        grades = []
        for i in range(config.n):
            if config.prompt_type == 'r1_zero' or config.prompt_type == 'r1_zero_three_shot_gsm8k':
                rewards = drgrpo_grader.r1_zero_reward_fn(finished['continuations'][i].text, final_answer, fast=False)
            elif config.prompt_type == 'question_only':
                rewards = drgrpo_grader.question_only_reward_fn(finished['continuations'][i].text, final_answer, fast=False)
            # contains format_reward, answer_reward, reward
            rewards['stopped'] = float(int(finished['continuations'][i].finish_reason == 'stop'))
            rewards['ans_len'] = len(finished['continuations'][i].token_ids)
            rewards['raw_text'] = finished['continuations'][i].text
            rewards['gt_ans'] = final_answer
            grades.append(rewards)
        finished['grades'] = grades
        del finished['continuations']
        
        pass1 = {}
        passn = {}
        for k in ['reward', 'answer_reward', 'format_reward', 'stopped', 'ans_len']:
            pass1[k] = sum([item[k] for item in grades]) / len(grades)
            passn[k] = max([item[k] for item in grades])
            avg_pass1[k] = (avg_pass1.get(k, 0.0) * now_cnt + pass1[k]) / (now_cnt + 1)
            avg_passn[k] = (avg_passn.get(k, 0.0) * now_cnt + passn[k]) / (now_cnt + 1)
            print(f'{k:20}: now pass1 = {pass1[k]:5.2f} tot pass1 = {avg_pass1[k]:5.2f} now passn = {passn[k]:5.2f} tot passn = {avg_passn[k]:5.2f}')
        finished['pass1'] = pass1
        finished['passn'] = passn

        now_cnt += 1
        all_finished.append(finished)
        wandb.log({'pass1': pass1, 'passn': passn, 'avg_pass1': avg_pass1, 'avg_passn': avg_passn, 'time_spent': time.time() - stime}, step=now_cnt)

        if config.first_n_samp: 
            if now_cnt > config.first_n_samp: break

    if config.backend == 'vllm':
        vllm_utils.stop_server(serv_proc)
    
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    with open(os.path.join(config.save_dir, run_name + '.jsonl'), 'w') as f:
        for fin in all_finished:
            json.dump(fin, f)
    with open(os.path.join(config.save_dir, run_name + '.summary.log'), 'w') as f:
        json.dump({'avg_pass1': avg_pass1, 'avg_passn': avg_passn}, f, indent=2)
    print('Done. Output to', os.path.join(config.save_dir, run_name + '.jsonl'), 'and', os.path.join(config.save_dir, run_name + '.summary.log'))
        

    