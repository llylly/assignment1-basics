uv run python -c "import torch; torch.cuda.empty_cache()"
uv run alignment/gsm8k_eval.py --backend vllm --prompt_type r1_zero
uv run python -c "import torch; torch.cuda.empty_cache()"
sleep 10
uv run alignment/gsm8k_eval.py --backend vllm --prompt_type r1_zero_three_shot_gsm8k
uv run python -c "import torch; torch.cuda.empty_cache()"
sleep 10
uv run alignment/gsm8k_eval.py --backend vllm --prompt_type question_only
uv run python -c "import torch; torch.cuda.empty_cache()"
sleep 10
uv run alignment/gsm8k_eval.py --backend native --prompt_type r1_zero
uv run python -c "import torch; torch.cuda.empty_cache()"
sleep 10
uv run alignment/gsm8k_eval.py --backend native --prompt_type r1_zero_three_shot_gsm8k
uv run python -c "import torch; torch.cuda.empty_cache()"
sleep 10
uv run alignment/gsm8k_eval.py --backend native --prompt_type question_only
uv run python -c "import torch; torch.cuda.empty_cache()"