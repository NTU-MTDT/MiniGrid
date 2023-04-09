conda create -n minigrid python=3.8
conda activate minigrid
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers
pip install gym-minigrid==1.1.0
pip install jupyter