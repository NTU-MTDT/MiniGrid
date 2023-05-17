from matplotlib import animation
import matplotlib.pyplot as plt
import random
import numpy
import torch
import os


def create_folder_if_necessary(path):
    if not os.path.exists(path):
        os.mkdir(path)


def fix_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def display_frames_as_gif(frames, name="output.gif"):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
    anim.save(name, fps=10)


def render_gif(env, seed, act_idxs, name="output.gif", print_reward=False):
    obs = env.reset(seed=seed)  # This now produces an RGB tensor only

    idxs = act_idxs
    idx2act = {0: 0, 1: 1, 2: 2}
    i = 0
    frames = []
    done = False
    while not done:
        if i >= len(idxs):
            break
        action = idx2act[idxs[i]]
        i += 1

        obs, reward, done, info = env.step(action)
        frames.append(env.render(mode="rgb_array"))
        if print_reward:
            print(reward)
    env.close()

    display_frames_as_gif(frames, name=name)
