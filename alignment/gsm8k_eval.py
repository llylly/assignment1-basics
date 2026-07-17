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
    vllm_server_no_host: bool = False # in this case, assume the host is already there
    vllm_ip: str = '127.0.0.1'
    vllm_port: int = 8080

def gsm8k_eval(eval_config: EvalConfig, dump_file=True, launch_wandb=True, verbose=True):

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
    with open(os.path.join(eval_config.data_dir, 'test.jsonl'), 'r') as f:
        test_data = [json.loads(item) for item in f.readlines()]
    
    if verbose: print('Loading prompt...')
    with open(os.path.join('alignment/prompts', eval_config.prompt_type + '.prompt'), 'r') as f:
        prompt_template = f.read()

    if verbose: print('Loading model...')
    if eval_config.backend == 'vllm':
        if not eval_config.vllm_server_no_host:
            serv_proc = vllm_utils.start_server(eval_config.model_dir, eval_config.vllm_ip, eval_config.vllm_port, eval_config.dtype, 0, 42, "auto", 'INFO')
            vllm_utils.wait_for_server(f'http://{eval_config.vllm_ip}:{eval_config.vllm_port}', serv_proc, 60)
    elif eval_config.backend == 'native':
        model, tokenizer = lmodeling_olmo.from_pretrained(eval_config.model_dir, eval_config.dtype)


    all_finished = []
    now_cnt = 0
    
    avg_pass1 = {}
    avg_passn = {}

    try:
        for item in tqdm.tqdm(test_data if eval_config.first_n_samp is None else test_data[:eval_config.first_n_samp]):
            raw_question = item['question']
            final_answer = item['answer'][item['answer'].find('####') + 4:].strip()
            templated_question = prompt_template.format(question=raw_question)
            continuations = []
            while len(continuations) < eval_config.n:
                amounts_to_gen = min(eval_config.n - len(continuations), eval_config.batch_size)
                if eval_config.backend == 'vllm':
                    ret = vllm_utils.generate_completions(f'http://{eval_config.vllm_ip}:{eval_config.vllm_port}', eval_config.model_dir, [templated_question], {
                        'temperature': eval_config.temperature,
                        'max_tokens': eval_config.max_new_tokens,
                        'n': amounts_to_gen,
                        'seed': 42,
                        'stop': ['</answer>'],
                        'include_stop_str_in_output': True,
                    }, None)
                elif eval_config.backend == 'native':
                    ret = generate(model, [templated_question for _ in range(amounts_to_gen)], tokenizer, eval_config.max_new_tokens, eval_config.temperature, extra_stop_tokens=['</answer>'], include_stop_str_in_output=True, verbose=False)
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
            if eval_config.prompt_type == 'r1_zero' or eval_config.prompt_type == 'r1_zero_three_shot_gsm8k':
                grade_fn = drgrpo_grader.r1_zero_reward_fn
            elif eval_config.prompt_type == 'question_only':
                grade_fn = drgrpo_grader.question_only_reward_fn

            for i in range(eval_config.n):
                # too slow
                # p = multiprocessing.get_context('spawn').Process(target=grade_fn, args=(finished['continuations'][i].text, final_answer, False))
                # p.start()
                # try:
                #     p.join(timeout=3)
                #     assert (p.exitcode is not None) and p.exitcode == 0 # first check whether it can safely exit
                #     rewards = grade_fn(finished['continuations'][i].text, final_answer, False)
                # except Exception:
                #     rewards = grade_fn(finished['continuations'][i].text, final_answer, True)
                # p.close()
                rewards = grade_fn(finished['continuations'][i].text, final_answer, False)

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
                print(f'{k:20}: now pass1 = {pass1[k]:5.3f} tot pass1 = {avg_pass1[k]:5.3f} now passn = {passn[k]:5.3f} tot passn = {avg_passn[k]:5.3f}')
            finished['pass1'] = pass1
            finished['passn'] = passn

            now_cnt += 1
            all_finished.append(finished)
            if launch_wandb:
                wandb.log({'pass1': pass1, 'passn': passn, 'avg_pass1': avg_pass1, 'avg_passn': avg_passn, 'time_spent': time.time() - stime}, step=now_cnt)
    
    except KeyboardInterrupt:
        if eval_config.backend == 'vllm' and not eval_config.vllm_server_no_host:
            vllm_utils.stop_server(serv_proc)
        return
    
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

    print('Done.')

    # sanitize
    resource.setrlimit(resource.RLIMIT_STACK, (soft_rlimit, hard_rlimit))

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
    gsm8k_eval(config, dump_file=True, launch_wandb=True, verbose=True)

    