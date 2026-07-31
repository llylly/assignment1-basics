# Linyi's Change

Now it's a more general codebase.

TODO:
[ ] Interleave RLVR generation and traning with bounded buffer.
[ ] SFT and DPO support.
[ ] Support Muon optimizer.
[ ] Support LoRA.
[ ] Basic Triton-based optimization.
[ ] Distributed training.
[ ] Support loading and training Qwen3.5 DeltaNet and Kimi KDA models.
[ ] Mechanistic interpretability - start from attention visualization figure for example
[ ] Support common pretraining benchmarks.
[ ] Support common model merging benchmarks.
[ ] Support common RLVR & PPO tasks.

Setup guide:
- upload data/
- uv sync --extra gpu
- set -e LD_LIBRARY_PATH or unset LD_LIBRARY_PATH
- uv run hf download allenai/OLMo-2-0425-1B --local-dir models/OLMo-2-0425-1B

# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

