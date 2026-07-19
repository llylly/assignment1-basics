from alignment import vllm_utils

# MODEL_DIR = 'models/OLMo-2-0425-1B'
# MODEL_DIR = 'models/SmolLM-1.7B'
MODEL_DIR = 'models/Qwen3.5-0.8B-Base'

if __name__ == '__main__':
    serv_proc = vllm_utils.start_server(MODEL_DIR, '127.0.0.1', 8080, 'bfloat16', 0, 42, "auto", 'INFO')
    vllm_utils.wait_for_server('http://127.0.0.1:8080', serv_proc, 300)
    try:
        while True:
            prompt = input('prompt: ')
            ret = vllm_utils.generate_completions('http://127.0.0.1:8080', MODEL_DIR, [prompt], {
                'temperature': 0,
                'max_tokens': 512,
                'n': 1,
                'seed': 42 
            }, 1)
            print(ret[0].text, 'Finish reason=', ret[0].finish_reason)
    except Exception:
        print('exited')
    vllm_utils.stop_server(serv_proc)