import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.twotask import EmptyEnv
from utils import *
import os
import time
import random
import numpy as np
import json
import itertools
from tqdm.auto import tqdm

env = EmptyEnv()
tasks_action_seq = env.subtasks
num_tasks = env.reward_dimension


def gen_traj(seed=0, weights=[1 for _ in range(num_tasks)]):
    env.reset(seed=seed)
    action_seq = ""
    rtg = np.zeros(num_tasks)
    while True:
        action_candidates = [str(random.randint(0, 2))] + tasks_action_seq
        action = random.choices(action_candidates, weights=weights)[0]
        # print(action)
        action_seq += action
        for a in action:
            obs, reward, done, info = env.step(int(a))
            rtg += np.array(reward)
            if done:
                break
        if done:
            break
    return action_seq, rtg


traj_per_product = 5
data = []
for combination in itertools.product(range(0, 11), repeat=num_tasks):
    combination = list(combination)
    if sum(combination) != 10:
        continue
    print(combination)
    for _ in range(traj_per_product):
        seed = random.randint(0, 10000)
        action_seq, rtg = gen_traj(seed=seed, weights=combination)
        data.append(
            {"seed": seed, "actions": [int(a) for a in action_seq], "rtg": rtg.tolist()}
        )

with open("dataset.json", "w") as f:
    json.dump(data, f, indent=4)

exit()

# preview
if os.path.exists("traj"):
    os.system("rm -rf traj")
os.mkdir("traj")

print("Generating preview")
env = EmptyEnv(render_mode="rgb_array")
for d in tqdm(random.sample(data, 10)):
    render_gif(
        env=env,
        seed=d["seed"],
        act_idxs=d["actions"],
        name=f"traj/seed_{d['seed']}.gif",
    )
