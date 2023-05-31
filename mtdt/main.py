import sys

sys.path.append("..")

import pickle
import random

import environments
import numpy as np
import torch
from decision_transformer.models.decision_transformer_1122 import DecisionTransformer
from minigrid.wrappers import *
from parameters import *
from tqdm.auto import tqdm
from utils import fix_seed

np.set_printoptions(edgeitems=30, linewidth=100000)

fix_seed(0)
env = gym.make(config["general"]["env_name"], render_mode="rgb_array")
obs, _ = env.reset()

state_dim = env.agent_view_size * env.agent_view_size * 3
return_dim = env.reward_dimension
act_dim = env.action_space.n

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    return_dim=return_dim,
    max_length=max_length,
    max_ep_len=max_ep_len,
    hidden_size=config["model"]["embed_dim"],
    n_layer=config["model"]["n_layer"],
    n_head=config["model"]["n_head"],
    n_inner=4 * config["model"]["embed_dim"],
    activation_function=config["model"]["activation_function"],
    n_positions=1024,
    resid_pdrop=config["model"]["dropout"],
    attn_pdrop=config["model"]["dropout"],
)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def load_trajectories(data_path):
    print("Loading trajectories...")
    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    states, traj_lens, returns, rtgs = [], [], [], []
    for i, path in enumerate(trajectories):
        states.append(np.array(path["observations"]).flatten())
        traj_lens.append(len(path["observations"]))
        returns.append(np.sum(np.array(path["rewards"]), axis=0))
        # trajectories[i]['rewards'] = path['rewards'][:, return_idx: return_idx + 1]

    states = np.array(states, dtype="object")
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # returns = returns[:, return_idx: return_idx + 1]

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    rtg_mean, rtg_std = np.mean(returns, axis=0), np.std(returns, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    num_eval_episodes = 1
    pct_traj = 1.0

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(np.array(returns)[:, 0])  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    print("Trajectories loaded")
    return {
        "trajectories": trajectories,
        "num_trajectories": num_trajectories,
        "num_timesteps": num_timesteps,
        "state_mean": state_mean,
        "state_std": state_std,
        "rtg_mean": rtg_mean,
        "rtg_std": rtg_std,
        "sorted_inds": sorted_inds,
        "p_sample": p_sample,
    }


trajectories_info = load_trajectories(data_path=config["general"]["data_path"])
trajectories = trajectories_info["trajectories"]
num_trajectories = trajectories_info["num_trajectories"]
num_timesteps = trajectories_info["num_timesteps"]
state_mean = trajectories_info["state_mean"]
state_std = trajectories_info["state_std"]
rtg_mean = trajectories_info["rtg_mean"]
rtg_std = trajectories_info["rtg_std"]
sorted_inds = trajectories_info["sorted_inds"]
p_sample = trajectories_info["p_sample"]

print(f"Number of timesteps: {num_timesteps}")
print(f"rtg_mean {rtg_mean}")
print(f"rtg_std {rtg_std}")


def get_batch(batch_size=256, max_len=max_length):
    batch_indices = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    # state, action, reward, done, return
    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_indices[i]])]
        start_index = random.randint(0, traj["rewards"].shape[0] - 1)  # start idx

        # get sequences from dataset
        # s.append(traj['observations'][si:si + max_len].r eshape(1, -1, *state_dim))
        s.append(
            traj["observations"][start_index : start_index + max_len].reshape(
                1, -1, state_dim
            )
        )
        a.append(
            traj["actions"][start_index : start_index + max_len].reshape(1, -1, act_dim)
        )
        r.append(
            traj["rewards"][start_index : start_index + max_len].reshape(
                1, -1, return_dim
            )
        )
        if "terminals" in traj:
            d.append(
                traj["terminals"][start_index : start_index + max_len].reshape(1, -1)
            )
        else:
            d.append(traj["dones"][start_index : start_index + max_len].reshape(1, -1))
        timesteps.append(
            np.arange(start_index, start_index + s[-1].shape[1]).reshape(1, -1)
        )
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
        rtg.append(
            discount_cumsum(traj["rewards"][start_index:], gamma=1.0)[
                : s[-1].shape[1] + 1
            ].reshape(1, -1, return_dim)
        )

        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, return_dim))], axis=1)

        # padding and state + reward normalization
        traj_len = s[-1].shape[1]
        # s[-1] = np.concatenate([np.zeros((1, max_len - tlen, *state_dim)), s[-1]], axis=1)
        s[-1] = np.concatenate(
            [np.zeros((1, max_len - traj_len, state_dim)), s[-1]], axis=1
        )
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate(
            [np.ones((1, max_len - traj_len, act_dim)) * -10.0, a[-1]], axis=1
        )
        r[-1] = np.concatenate(
            [np.zeros((1, max_len - traj_len, return_dim)), r[-1]], axis=1
        )
        d[-1] = np.concatenate([np.ones((1, max_len - traj_len)) * 2, d[-1]], axis=1)
        rtg[-1] = (
            np.concatenate(
                [np.zeros((1, max_len - traj_len, return_dim)), rtg[-1]], axis=1
            )
            / scale
        )

        mask_value = -100
        mask_dims = np.random.choice(
            np.arange(rtg[-1].shape[2]),
            size=random.randint(0, rtg[-1].shape[2] - 1),
            replace=False,
        )
        # rtg[-1][:, :, mask_dims] = mask_value

        timesteps[-1] = np.concatenate(
            [np.zeros((1, max_len - traj_len)), timesteps[-1]], axis=1
        )
        mask.append(
            np.concatenate(
                [np.zeros((1, max_len - traj_len)), np.ones((1, traj_len))], axis=1
            )
        )
    s = torch.from_numpy(np.concatenate(s, axis=0)).to(
        dtype=torch.float32, device=device
    )
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(
        dtype=torch.float32, device=device
    )
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
        dtype=torch.float32, device=device
    )
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(
        dtype=torch.float32, device=device
    )
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
        dtype=torch.long, device=device
    )
    # print(r.shape)
    # print(rtg.shape)
    
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    ### normalize rtg
    # rtg = (rtg - rtg_mean) / rtg_std

    ### random mask some dimensions of the rtg
    # mask_value = -100
    # rtg = rtg.permute(0, 2, 1)
    # ### choose random dimensions to mask
    # mask_dims = np.random.choice(np.arange(rtg.shape[1]), size=random.randint(0, rtg.shape[1]-1) , replace=False)
    # rtg[:, mask_dims, :] = mask_value
    # rtg = rtg.permute(0, 2, 1)

    return s, a, r, d, rtg, timesteps, mask


idx2act = {0: 0, 1: 1, 2: 2}


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    return_dim,
    model,
    idx2act,
    seed=0,
    max_ep_len=1000,
    scale=1.0,
    state_mean=0.0,
    state_std=1.0,
    device=device,
    target_return=None,
    mode="normal",
    temperature=1.0,
    render=False,
):
    with torch.no_grad():
        model.eval()
        model.to(device=device)

        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset(seed=seed)[0]["image"]
        if mode == "noise":
            state = state + np.random.normal(0, 0.1, size=state.shape)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = (
            torch.from_numpy(state)
            .reshape(1, state_dim)
            .to(device=device, dtype=torch.float32)
        )
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        chosen_actions = []
        rewards = torch.zeros((0, return_dim), device=device, dtype=torch.float32)

        ep_return = target_return
        # target_return = torch.tensor(
        #     ep_return, device=device, dtype=torch.float32
        # ).reshape(1, return_dim)
        target_return = (
            ep_return.clone().detach().reshape(1, return_dim).to(device=device)
        )
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        sim_states = []
        frames = []

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):
            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=device)], dim=0
            )
            rewards = torch.cat(
                [rewards, torch.zeros((1, return_dim), device=device)], dim=0
            )

            # print(target_return)
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            # # get the max_action
            # action = np.argmax(action)
            # sample action by action logits
            action *= temperature
            action_probs = np.exp(action) / np.sum(np.exp(action))
            # print(action_probs)
            action = np.random.choice(np.arange(act_dim), p=action_probs)
            # action = np.argmax(action_probs)
            chosen_actions.append(action)

            state, reward, terminated, truncated, _ = env.step(idx2act[action])
            done = terminated or truncated
            if render:
                frames.append(env.render())
            state = state["image"]

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            reward = torch.tensor(reward).to(device)
            rewards[-1] = reward

            if mode != "delayed":
                pred_return = target_return[-1]
                ### only update those dimensions that are > 0
                mask = (pred_return > 0).float()
                pred_return = pred_return - reward / scale * mask
                # pred_return = target_return[-1] - reward
            else:
                pred_return = target_return[0, -1]

            target_return = torch.cat(
                [target_return, pred_return.reshape(1, return_dim)], dim=0
            )
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1),
                ],
                dim=1,
            )

            episode_return += reward * 1.0
            episode_length += 1

            # if (episode_return.cpu() >= ep_return).all():
            #     done = True

            if done:
                break
        model.train()
    return episode_return, episode_length, chosen_actions, target_return[0], frames


def eval(model, env, rw, seed=0, temperature=1.0, render=False):
    target_return = torch.tensor(rw)
    result = evaluate_episode_rtg(
        idx2act=idx2act,
        state_dim=state_dim,
        act_dim=act_dim,
        return_dim=return_dim,
        model=model,
        env=env,
        device=device,
        target_return=target_return,
        mode="normal",
        max_ep_len=max_ep_len,
        scale=torch.tensor(scale).to(device=device),
        state_mean=np.array(state_mean),
        state_std=np.array(state_std),
        seed=seed,
        temperature=temperature,
        render=render,
    )
    return {
        "actual_return": result[0].cpu().numpy(),
        "episode_length": result[1],
        "actions": np.array(result[2]),
        "target_return": result[3].cpu().numpy(),
        "frames": np.array(result[4]),
    }


def eval_n(model, env, n=1, datapoints_per_task=5, temperature=1.0):
    return_errors = []
    frames = []
    for task_id in tqdm(range(return_dim), desc="task", leave=False):
        return_error_temp = []
        frames_temp = []

        stds = np.array(
            [
                -1 + 2.0 * (i / (datapoints_per_task - 1))
                for i in range(datapoints_per_task)
            ]
        )
        rtgs = rtg_mean[task_id] + stds * rtg_std[task_id]
        print(f"Evaluating task{task_id} with rtg = {rtgs}")

        for seed in tqdm(range(10000, 10000 + n), desc="seed", leave=False):
            for i in range(datapoints_per_task):
                target_return = [
                    rtgs[i] if i == task_id else 0 for i in range(return_dim)
                ]
                render = i == datapoints_per_task // 2
                result = eval(
                    model=model,
                    rw=target_return,
                    env=env,
                    seed=seed,
                    temperature=temperature,
                    render=render,
                )
                if render:
                    frames_temp.append(result["frames"])
                actual_return = result["actual_return"]
                return_error = abs(target_return - actual_return)
                return_error_temp.append(return_error)
        return_errors.append(return_error_temp)
        frames.append(np.concatenate(frames_temp))

    return return_errors, frames


def eval_func(
    model,
    n=config["evaluation"]["seed_count"],
    dpt=config["evaluation"]["datapoints_per_task"],
    temperature=1.0,
):
    return_errors, frames = eval_n(
        model=model, env=env, n=n, datapoints_per_task=dpt, temperature=temperature
    )

    n = n * dpt
    error_mean = [np.mean(r) for r in return_errors]
    error_std = [np.std(r) for r in return_errors]
    error_confidence_interval = [1.96 * e / np.sqrt(n) for e in error_std]
    return {
        "error_mean": error_mean,
        "error_std": error_std,
        "error_confidence_interval": error_confidence_interval,
        "frames": frames,
    }


model = model.to(device=device)
