run_name = "three_task_test"

data_path = "../dataset/dataset.pkl"
device = "cuda:0"

state_dim = 7 * 7 * 3
act_dim = 3
return_dim = 3

config = {
    "train": {
        "num_steps_per_iter": 5000,
        "max_iterations": 50,
        "warmup_steps": 3000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_length": 5,
        "max_ep_len": 100,
        "scale": 1.0,
    },
    "evaluation": {
        "seed_count": 20,
        "datapoints_per_task": 5,
    },
    "model": {
        "embed_dim": 256,
        "n_layer": 4,
        "n_head": 2,
        "activation_function": "relu",
        "dropout": 0.1,
    },
}

max_ep_len = config["train"]["max_ep_len"]
max_length = config["train"]["max_length"]
scale = config["train"]["scale"]
max_iterations = config["train"]["max_iterations"]
num_steps_per_iter = config["train"]["num_steps_per_iter"]
