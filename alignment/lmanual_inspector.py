"""
    Nice print for inspection
    A handy tool
"""
import os
from pathlib import Path
import json

import drgrpo_grader

paths = [
    'eval/gsm8k_test_baseeval_vllm_prompt_question_only_temp_1.0_n_5_max_new_tokens_512_models-OLMo-2-0425-1B_bfloat16_bs_5__20260716_100619.jsonl',
    'eval/gsm8k_test_baseeval_vllm_prompt_r1_zero_three_shot_gsm8k_temp_1.0_n_5_max_new_tokens_512_models-OLMo-2-0425-1B_bfloat16_bs_5__20260716_041249.jsonl',
    'eval/gsm8k_test_baseeval_vllm_prompt_r1_zero_temp_1.0_n_5_max_new_tokens_512_models-OLMo-2-0425-1B_bfloat16_bs_5__20260716_035309.jsonl'
]

# print(drgrpo_grader.r1_zero_reward_fn(" Each robe takes 2 bolts in blue and 1 bolt in white. So in total, it takes 2 + 1 = 3 bolts. <answer> 3 </answer>", "3", False))

if __name__ == '__main__':
    for path in paths:
        print(path)
        case_cnt = 0
        with open(path, 'r') as f:
            for line in f.readlines():
                struct = json.loads(line)
                final_answer = struct['final_answer']
                for grade in struct['grades']:
                    if grade['format_reward'] == 1.0 and grade['answer_reward'] == 1.0:
                        print(f'\n***** Case #{case_cnt}:')
                        print(struct['question'])
                        print('GT answer:', final_answer)
                        print('======\nModel Response:\n')
                        print(grade['raw_text'])
                        case_cnt += 1
                        input('')
                        break
                if case_cnt >= 10:
                    break