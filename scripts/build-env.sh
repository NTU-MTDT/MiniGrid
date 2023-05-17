conda create -n minigrid python=3.9
conda activate minigrid

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers minigrid
pip install wandb moviepy imageio