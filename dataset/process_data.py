import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environments.twotask import EmptyEnv
import json
import pickle

# import os
# from envs.tripledoors import TripleDoorsEnv
# from envs.multiconstraints1115 import EmptyEnv
# from wrappers import (
#     FullyObsWrapper,
#     RGBImgObsWrapper,
#     RGBImgPartialObsWrapper,
#     ImgObsWrapper,
# )
from tqdm.auto import tqdm
import numpy as np

log_path = "./logs"

# def process_data(log_path):
#     log_files = os.listdir(log_path)
#     log_files = [os.path.join(log_path, f) for f in log_files]
#     log_files = [f for f in log_files if os.path.isfile(f)]
#     log_files = [f for f in log_files if f.endswith(".json")]
#     log_files = sorted(log_files, key=lambda f: int(os.path.basename(f).split(".")[0]))
#     print(f"Found {len(log_files)} log files")

#     data = []
#     for log_file in log_files:
#         with open(log_file, "r") as f:
#             print(log_file)
#             log = json.load(f)
#         data.append(log)

#     return data

# data = process_data(log_path)

with open("dataset.json", "r") as f:
    data = json.load(f)

### count data by rtg
counter = {}
for d in data:
    rtg = tuple(d["rtg"])
    counter[rtg] = counter.get(rtg, 0) + 1

# with open("counter.pkl", "wb") as f:
#     pickle.dump(counter, f)

with open("reward_counter.txt", "w") as f:
    for k, v in sorted(counter.items(), reverse=True):
        print(f"{k}: {v}", file=f)

### retrieve observation and reward
env = EmptyEnv(render_mode="rgb_array")
# env = RGBImgObsWrapper(env)
# env = ImgObsWrapper(env)


## act_to_idx = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}
### act_to_idx = {0: 0, 1: 1, 2: 2, 5: 3}
act_to_idx = {0: 0, 1: 1, 2: 2}
## suppress the dones
act_space = len(act_to_idx.keys())


def collect_experience(env, d):
    obs = env.reset(seed=d["seed"])
    exps = {"observations": [obs["image"]], "actions": [], "rewards": [], "dones": []}
    for i in d["actions"]:
        action = i
        obs, reward, done, _ = env.step(action)
        exps["observations"].append(obs["image"])
        exps["actions"].append(np.eye(act_space)[act_to_idx[action]])  # one-hot
        exps["rewards"].append(reward)
        exps["dones"].append(done)
        if done:
            break
        ## turn to numpy
    exps["observations"] = exps["observations"][:-1]
    exps["observations"], exps["actions"], exps["rewards"], exps["dones"] = (
        np.array(exps["observations"]),
        np.array(exps["actions"]),
        np.array(exps["rewards"]),
        np.array(exps["dones"]),
    )
    return exps


exps = []
total_timesteps = 0
for d in tqdm(data):
    exp = collect_experience(env, d)
    exps.append(exp)
    total_timesteps += exp["observations"].shape[0]
    # obs = exp["observations"]
    ### rotate observation 90, 180, 270 degrees
    # for i in range(3):
    #     obs = np.rot90(obs, axes=(1, 2))
    #     exp = {"observations": obs, "actions": exp["actions"], "rewards": exp["rewards"], "dones": exp["dones"]}
    #     exps.append(exp)

print(f"Total experiences: {len(exps)}")
print(f"Total timesteps: {total_timesteps}")
print(f"Observation shape: {exps[0]['observations'].shape}")

### save data
import pickle

with open("dataset.pkl", "wb") as f:
    pickle.dump(exps, f)
