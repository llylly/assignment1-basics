from dataclasses import dataclass, field, asdict
from typing import Literal
import resource
import os
import json
import tqdm
import math
import time
from datetime import datetime
import multiprocessing

import tyro
import wandb
from alignment import vllm_utils
import basics.lmodeling_olmo as lmodeling_olmo
from basics.linference import generate
from alignment import drgrpo_grader

@dataclass
class GSM8KEvalConfig:
    backend: Literal['native', 'vllm'] = 'native' # 'vllm' / 'native'
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16' # or 'float32'
    device: str = 'cuda'
    prompt_type: Literal['r1_zero', 'question_only', 'r1_zero_three_shot_gsm8k'] = 'r1_zero' # or 'question_only'
    max_new_tokens: int = 512
    temperature: float = 1.0
    n: int = 5
    batch_size: int = 5
    first_n_samp: int | None = None
    model_dir: str = 'models/OLMo-2-0425-1B'
    data_file: str = 'data/gsm8k/test.jsonl'
    run_suffix: str = 'baseeval'
    save_dir: str = 'eval'
    vllm_server_no_host: bool = False # in this case, assume the host is already there
    vllm_ip: str = '127.0.0.1'
    vllm_port: int = 8080

def gsm8k_question_formulator(sample: dict, prompt_template: str) -> str:
    return prompt_template.format(question=sample['question'])

def gsm8k_question_grader(response: str, ground_truth: str, prompt_type: Literal['r1_zero', 'question_only', 'r1_zero_three_shot_gsm8k']) -> dict:
    # potentially modified queston_sample, along with grade result which is a dict containing at least 'reward': float

    if prompt_type == 'r1_zero' or prompt_type == 'r1_zero_three_shot_gsm8k':
        grade_fn = drgrpo_grader.r1_zero_reward_fn
    elif prompt_type == 'question_only':
        grade_fn = drgrpo_grader.question_only_reward_fn
    return grade_fn(response, ground_truth, fast=False)

def gsm8k_seteval(eval_config: GSM8KEvalConfig, dump_file=True, launch_wandb=True, verbose=True):

    nowtime = datetime.now().strftime('_%Y%m%d_%H%M%S')
    run_name = f'gsm8k_test_{eval_config.run_suffix}_{eval_config.backend}_prompt_{eval_config.prompt_type}_temp_{eval_config.temperature}_n_{eval_config.n}_max_new_tokens_{eval_config.max_new_tokens}_{eval_config.model_dir.replace("/", "-")}_{eval_config.dtype}_bs_{eval_config.batch_size}_{nowtime}'

    # fuck antlr4
    soft_rlimit, hard_rlimit = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (min(1048576 * 1024, hard_rlimit), hard_rlimit))

    # wandb
    if launch_wandb:
        wandb.init(project='LLLM_eval_gsm8k_test', name=run_name, config=asdict(eval_config), dir=os.path.join(eval_config.save_dir, 'wandb_logs'))
    stime = time.time()

    # load data
    with open(eval_config.data_file, 'r') as f:
        test_data = [json.loads(item) for item in f.readlines()]
    
    if verbose: print('Loading prompt...')
    with open(os.path.join('alignment/prompts', eval_config.prompt_type + '.prompt'), 'r') as f:
        prompt_template = f.read()

    if verbose: print('Loading model...')
    if eval_config.backend == 'vllm':
        if not eval_config.vllm_server_no_host:
            serv_proc = vllm_utils.start_server(eval_config.model_dir, eval_config.vllm_ip, eval_config.vllm_port, eval_config.dtype, int(eval_config.device[5:]) if len(eval_config.device) > 4 else 0, None, "auto", 'INFO')
            vllm_utils.wait_for_server(f'http://{eval_config.vllm_ip}:{eval_config.vllm_port}', serv_proc, 300)
    elif eval_config.backend == 'native':
        model, tokenizer = lmodeling_olmo.from_pretrained(eval_config.model_dir, eval_config.dtype, eval_config.device, flash_attn=False) # for gsm8k, I don't see much benefit of using flash attn


    all_finished = []
    
    avg_pass1 = {}
    avg_passn = {}

    try:
        # new version with continuous batching
        test_data = test_data if eval_config.first_n_samp is None else test_data[:eval_config.first_n_samp]
        continuations = []
        for i in tqdm.tqdm(range(0, len(test_data) * eval_config.n, eval_config.batch_size)):
            prompts = [gsm8k_question_formulator(test_data[j // eval_config.n], prompt_template) for j in range(i, min(i+eval_config.batch_size, len(test_data) * eval_config.n))]

            if eval_config.backend == 'vllm':
                ret = vllm_utils.generate_completions(f'http://{eval_config.vllm_ip}:{eval_config.vllm_port}', eval_config.model_dir, prompts, {
                    'temperature': eval_config.temperature,
                    'max_tokens': eval_config.max_new_tokens,
                    'n': 1,
                    'seed': None,
                    'stop': ['</answer>'],
                    'include_stop_str_in_output': True,
                }, None)
            elif eval_config.backend == 'native':
                ret = generate(model, prompts, tokenizer, eval_config.max_new_tokens, eval_config.temperature, device=eval_config.device, extra_stop_tokens=['</answer>'], include_stop_str_in_output=True, verbose=False)
            else:
                raise NotImplementedError
            continuations.extend(ret)

            old_q_num = i // eval_config.n
            new_q_num = min(i + eval_config.batch_size, len(test_data) * eval_config.n) // eval_config.n

            for q_idx in range(old_q_num, new_q_num):
                finished = {
                    'question': test_data[q_idx]['question'],
                    'prompt': gsm8k_question_formulator(test_data[q_idx], prompt_template),
                    'answer': test_data[q_idx]['answer'],
                    'continuations': continuations[q_idx * eval_config.n: (q_idx + 1) * eval_config.n]
                }
                grades = []

                for j in range(eval_config.n):
                    final_answer = finished['answer'][finished['answer'].find('####') + 4:].strip()
                    finished['final_answer'] = final_answer
                    rewards = gsm8k_question_grader(finished['continuations'][j].text, final_answer, eval_config.prompt_type)

                    # contains format_reward, answer_reward, reward
                    rewards['stopped'] = float(int(finished['continuations'][j].finish_reason == 'stop'))
                    rewards['ans_len'] = len(finished['continuations'][j].token_ids)
                    rewards['raw_text'] = finished['continuations'][j].text
                    rewards['gt_ans'] = finished['final_answer']
                    grades.append(rewards)

                finished['grades'] = grades
                del finished['continuations']
                
                pass1 = {}
                passn = {}
                for k in ['reward', 'answer_reward', 'format_reward', 'stopped', 'ans_len']:
                    pass1[k] = sum([item[k] for item in grades]) / len(grades)
                    passn[k] = max([item[k] for item in grades])
                    avg_pass1[k] = (avg_pass1.get(k, 0.0) * q_idx + pass1[k]) / (q_idx + 1)
                    avg_passn[k] = (avg_passn.get(k, 0.0) * q_idx + passn[k]) / (q_idx + 1)
                    if verbose: print(f'{k:20}: now pass1 = {pass1[k]:5.3f} tot pass1 = {avg_pass1[k]:5.3f} now passn = {passn[k]:5.3f} tot passn = {avg_passn[k]:5.3f}')
                finished['pass1'] = pass1
                finished['passn'] = passn

                all_finished.append(finished)
                if launch_wandb:
                    wandb.log({'pass1': pass1, 'passn': passn, 'avg_pass1': avg_pass1, 'avg_passn': avg_passn, 'time_spent': time.time() - stime}, step=q_idx)

    finally:
        if eval_config.backend == 'vllm' and not eval_config.vllm_server_no_host:
            vllm_utils.stop_server(serv_proc)
    
    if dump_file:
        if not os.path.exists(eval_config.save_dir):
            os.makedirs(eval_config.save_dir)
        with open(os.path.join(eval_config.save_dir, run_name + '.jsonl'), 'w') as f:
            for fin in all_finished:
                json.dump(fin, f)
                print('', file=f) # add \n
        with open(os.path.join(eval_config.save_dir, run_name + '.summary.log'), 'w') as f:
            json.dump({'avg_pass1': avg_pass1, 'avg_passn': avg_passn}, f, indent=2)
        with open(os.path.join(eval_config.save_dir, run_name + '.config.log'), 'w') as f:
            json.dump(asdict(eval_config), f, indent=2)

        print('Saved to')
        print(os.path.join(eval_config.save_dir, run_name + '.jsonl'))
        print(os.path.join(eval_config.save_dir, run_name + '.summary.log'))
        print(os.path.join(eval_config.save_dir, run_name + '.config.log'))

    # sanitize
    resource.setrlimit(resource.RLIMIT_STACK, (soft_rlimit, hard_rlimit))

    return {'avg_pass1': avg_pass1, 'avg_passn': avg_passn}, all_finished

"""
Example commands:
uv run alignment/benchmarks/lgsm8k_eval.py --backend vllm --prompt_type r1_zero
uv run alignment/benchmarks/lgsm8k_eval.py --backend vllm --prompt_type r1_zero_three_shot_gsm8k
uv run alignment/benchmarks/lgsm8k_eval.py --backend vllm --prompt_type question_only
uv run alignment/benchmarks/lgsm8k_eval.py --backend native --prompt_type r1_zero
uv run alignment/benchmarks/lgsm8k_eval.py --backend native --prompt_type r1_zero_three_shot_gsm8k
uv run alignment/benchmarks/lgsm8k_eval.py --backend native --prompt_type question_only
uv run alignment/benchmarks/lgsm8k_eval.py --backend vllm --prompt_type r1_zero --run-suffix rleval --model-dir models/rl/olmo2_1B_gsm8k/base_rl_r1zero_20260727_141916/hf_ckpts/step_0000199 --batch-size 256
uv run alignment/benchmarks/lgsm8k_eval.py --backend native --prompt_type r1_zero --run-suffix rleval --model-dir models/rl/olmo2_1B_gsm8k/base_rl_r1zero_20260727_141916/hf_ckpts/step_0000199 --batch-size 192 --device cuda:1
"""
if __name__ == '__main__':
    config = tyro.cli(GSM8KEvalConfig)
    gsm8k_seteval(config, dump_file=True, launch_wandb=True, verbose=True)

    