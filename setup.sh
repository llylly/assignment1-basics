apt update
apt install nvtop
apt install btop
apt install htop
apt install fish
apt install tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
hf download allenai/OLMo-2-0425-1B --local-dir models/OLMo-2-0425-1B
git config --global user.email "linyi@sfu.ca"
git config --global user.name "Linyi Li"