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

def get_trainable_task_grader(task_name: str) -> Callable:
    assert task_name in trainable_tasks
    if task_name == 'gsm8k':
        from alignment.benchmarks import lgsm8k_eval
        return lgsm8k_eval.gsm8k_question_grader
    else:
        raise NotImplementedError

def get_testable_task_grader(task_name: str) -> Tuple[object, Callable[[object, bool, bool, bool], None]]:
    assert task_name in testable_tasks
    if task_name == 'gsm8k':
        from alignment.benchmarks import lgsm8k_eval
        return lgsm8k_eval.GSM8KEvalConfig, lgsm8k_eval.gsm8k_seteval
    else:
        raise NotImplementedError
