import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environments.threetask import EmptyEnv
from mtdt.utils import *
import time
import random
import numpy as np
import json
from tqdm.auto import tqdm

movement = {
    0: np.array([1, 0]),  # E
    1: np.array([0, 1]),  # S
    2: np.array([-1, 0]),  # W
    3: np.array([0, -1]),  # N
}


def turnto(agent_dir, target_dir):
    if agent_dir == target_dir:
        return []
    elif (agent_dir + 1) % 4 == target_dir:
        return [1]
    elif (agent_dir + 2) % 4 == target_dir:
        return random.choice([[0, 0], [1, 1]])
    elif (agent_dir + 3) % 4 == target_dir:
        return [0]


env = EmptyEnv()
random_act_candidates = env.subtasks
num_tasks = env.reward_dimension


def gen_traj(task, seed=0, p=0.5):
    env.reset(seed=seed)
    action_seq = ""
    goal_pos = np.array(env.goal_pos)
    # rtg = np.array([0.0 for _ in range(num_tasks)])
    rtg = np.zeros(num_tasks)
    # task = np.random.randint(0, num_tasks)
    done = False
    if task == 0:
        while not done:
            full_obs = env.grid.encode().transpose(2, 1, 0)
            agent_pos = env.agent_pos
            agent_dir = env.agent_dir
            if random.random() > p:
                action = str(random.randint(0, 2))
            else:
                diff = goal_pos - agent_pos
                next_move = []
                if diff[0] > 0:
                    next_move.append(0)
                elif diff[0] < 0:
                    next_move.append(2)
                if diff[1] > 0:
                    next_move.append(1)
                elif diff[1] < 0:
                    next_move.append(3)
                random.shuffle(next_move)
                next_move = next_move[0]
                action = "".join([str(i) for i in (turnto(agent_dir, next_move) + [2])])
            action_seq += action
            for a in action:
                obs, reward, done, info = env.step(int(a))
                rtg += np.array(reward)
                if done:
                    break
    else:
        while not done:
            if random.random() < p:
                action = random_act_candidates[task - 1]
            else:
                action = str(random.randint(0, 2))
            action_seq += action
            for a in action:
                obs, reward, done, info = env.step(int(a))
                rtg += np.array(reward)
                if done:
                    break
    return action_seq, rtg


data = []


def generate_data(total_seeds=1, traj_per_seed=1, p=0.5):
    for seed in tqdm(range(total_seeds)):
        for _ in range(traj_per_seed):
            for task in range(num_tasks):
                action_seq, rtg = gen_traj(task=task, seed=seed, p=p)
                data.append(
                    {
                        "task": task,
                        "p": p,
                        "seed": seed,
                        "actions": [int(a) for a in action_seq],
                        "rtg": rtg.tolist(),
                    }
                )


print("Expert")
generate_data(total_seeds=10, traj_per_seed=1, p=1)

print("Normal")
generate_data(total_seeds=20, traj_per_seed=10, p=0.7)
generate_data(total_seeds=20, traj_per_seed=10, p=0.3)

print("Bad")
generate_data(total_seeds=10, traj_per_seed=5, p=0.05)

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
