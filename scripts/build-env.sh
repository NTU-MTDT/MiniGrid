conda create -n Minigrid python=3.9
conda activate Minigrid

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers minigrid
pip install wandb moviepy imageio
pip install matplotlib