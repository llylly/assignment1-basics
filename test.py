from basics.lmodeling_olmo import from_pretrained
from basics.linference import generate
from alignment import vllm_utils

MODEL_DIR = 'models/OLMo-2-0425-1B'
# MODEL_DIR = 'models/SmolLM-1.7B'
# MODEL_DIR = 'models/Qwen3.5-0.8B-Base'

if __name__ == '__main__':
    model, tokenizer = from_pretrained('models/OLMo-2-0425-1B', dtype='bfloat16', flash_attn=True)

    ret = (generate(model, ["He", "He"], tokenizer=tokenizer, temperature=0.0, top_p=0.9, return_log_probs=True))

    print(len(ret[0].token_ids))
    print(ret[0].log_probs.numel())

    print(len(ret[1].token_ids))
    print(ret[1].log_probs.numel())

    print(ret[0].log_probs)


# if __name__ == '__main__':
#     serv_proc = vllm_utils.start_server(MODEL_DIR, '127.0.0.1', 8080, 'bfloat16', 2, 42, "auto", 'INFO')
#     vllm_utils.wait_for_server('http://127.0.0.1:8080', serv_proc, 300)
#     try:
#         while True:
#             prompt = input('prompt: ')
#             if len(prompt) == 0: break
#             ret = vllm_utils.generate_completions('http://127.0.0.1:8080', MODEL_DIR, [prompt], {
#                 'temperature': 0,
#                 'max_tokens': 512,
#                 'n': 1,
#                 'seed': 42 
#             }, 1, True)
#             print(ret[0].text, 'Finish reason=', ret[0].finish_reason, ret[0].log_probs)
#             print(len(ret[0].token_ids), len(ret[0].log_probs))
#         # except Exception as e:
#         print('exited')
#     finally:
#         vllm_utils.stop_server(serv_proc)