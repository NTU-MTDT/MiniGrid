run_name = "threetasks"

env_name = "ThreeTask-v0"
data_path = "/tmp2/B09901171/rl-starter-files/dataset/threetasks.pkl"
ckpt_path = "threetasks_ckpt"

device = "cuda:1"

config = {
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

max_ep_len = config["train"]["max_ep_len"]
max_length = config["train"]["max_length"]
scale = config["train"]["scale"]
max_iterations = config["train"]["max_iterations"]
num_steps_per_iter = config["train"]["num_steps_per_iter"]

assert config["evaluation"]["datapoints_per_task"] % 2 == 1
