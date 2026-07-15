from alignment import vllm_utils

if __name__ == '__main__':
    serv_proc = vllm_utils.start_server('models/OLMo-2-0425-1B', '127.0.0.1', 8080, 0, 42, "auto", 'INFO')
    vllm_utils.wait_for_server('http://127.0.0.1:8080', serv_proc, 60)
    try:
        while True:
            prompt = input('prompt: ')
            ret = vllm_utils.generate_completions('http://127.0.0.1:8080', 'models/OLMo-2-0425-1B', [prompt], {
                'temperature': 0,
                'max_tokens': 512,
                'n': 1,
                'seed': 42 
            }, 1)
            print(ret[0].text, 'Finish reason=', ret[0].finish_reason)
    except KeyboardInterrupt:
        print('exited')
    vllm_utils.stop_server(serv_proc)