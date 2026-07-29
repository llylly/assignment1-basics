from typing import Callable, Tuple
from dataclasses import dataclass

trainable_tasks = [
    'gsm8k'
    # add new tasks here   
]

testable_tasks = [
    'gsm8k'
    # add new tasks here
]

def get_question_formulator(task_name: str) -> Callable[[dict, str], str]: # sample, prompt_template -> prompt
    assert task_name in trainable_tasks
    if task_name == 'gsm8k':
        from alignment.benchmarks import lgsm8k_eval
        return lgsm8k_eval.gsm8k_question_formulator
    else:
        raise NotImplementedError

def get_task_grader(task_name: str, **kwargs) -> Callable[[str, str], dict]: # response, ground_truth -> rewards
    assert task_name in trainable_tasks
    if task_name == 'gsm8k':
        from alignment.benchmarks import lgsm8k_eval
        prompt_type = kwargs['prompt_type']
        return lambda x, y: lgsm8k_eval.gsm8k_question_grader(x, y, prompt_type=prompt_type)
    else:
        raise NotImplementedError

def get_testable_task_setgrader(task_name: str) -> Tuple[object, Callable[[object, bool, bool, bool, object | None, object | None], Tuple[dict, list]]]:
    assert task_name in testable_tasks
    if task_name == 'gsm8k':
        from alignment.benchmarks import lgsm8k_eval
        return lgsm8k_eval.GSM8KEvalConfig, lgsm8k_eval.gsm8k_seteval
    else:
        raise NotImplementedError
