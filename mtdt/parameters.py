config = {
    "general": {
        "run_name": "LavaGoal",
        # "run_name": None,
        "env_name": "LavaGoal-v0",
        "data_path": "/home/ray/project/minigrid-online/dataset/lava_goal/lava_goal.pkl",
        "ckpt_path": "LavaGoal_ckpt",
        "device": "cuda:1",
    },
    "train": {
        "num_steps_per_iter": 5000,
        "max_iterations": 100,
        "warmup_steps": 3000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_length": 20,
        "max_ep_len": 100,
        "scale": 1.0,
    },
    "evaluation": {
        "seed_count": 10,
        "datapoints_per_task": 3,
    },
    "model": {
        "embed_dim": 256,
        "n_layer": 4,
        "n_head": 4,
        "activation_function": "relu",
        "dropout": 0.1,
    },
}

device = config["general"]["device"]
max_ep_len = config["train"]["max_ep_len"]
max_length = config["train"]["max_length"]
scale = config["train"]["scale"]
max_iterations = config["train"]["max_iterations"]
num_steps_per_iter = config["train"]["num_steps_per_iter"]

assert config["evaluation"]["datapoints_per_task"] % 2 == 1
