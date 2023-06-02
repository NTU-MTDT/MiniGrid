from main import *
from utils import create_folder_if_necessary
import os
import wandb
from torch.nn import CrossEntropyLoss
from decision_transformer.training.seq_trainer import SequenceTrainer

log_to_wandb = config["general"]["run_name"] is not None
if log_to_wandb:
    wandb.init(
        entity="mtdt", project="MTDT", name=config["general"]["run_name"], config=config
    )
    wandb.define_metric("training/train_loss_mean", summary="min")
    wandb.define_metric("training/action_error", summary="min")

print(f"max_iterations {max_iterations}")
print(f"num_steps_per_iter {num_steps_per_iter}")

ckpt_path = config["general"]["ckpt_path"]
create_folder_if_necessary(ckpt_path)

os.system(f"cp parameters.py {ckpt_path}/parameters.py")
os.system(f"cp {config['general']['data_path']} {ckpt_path}/data.pkl")

model.train()
model.requires_grad_(True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["train"]["learning_rate"],
    weight_decay=config["train"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda steps: min((steps + 1) / config["train"]["weight_decay"], 1)
)
loss_a = CrossEntropyLoss()
trainer = SequenceTrainer(
    model=model,
    optimizer=optimizer,
    batch_size=config["train"]["batch_size"],
    get_batch=get_batch,
    scheduler=scheduler,
    # cross entropy loss
    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: loss_a(a_hat, a),
    eval_fns=[eval_func],
)


for iter_num in range(max_iterations):
    output = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iter_num)
    print("=" * 80)
    print(f"Iteration {iter_num}")
    for k, v in output.items():
        if "frame" not in k:
            print(f"{k}: {v}")

    keys_to_expand = [
        "evaluation/error_mean",
        "evaluation/error_std",
        "evaluation/error_confidence_interval",
    ]
    for key in keys_to_expand:
        for i, k in enumerate(output[key]):
            output[f"{key}_{i}"] = output[key][i]
        output.pop(key)

    for i, k in enumerate(output["evaluation/frames"]):
        frames = output["evaluation/frames"][i]
        frames = frames.transpose((0, 3, 1, 2))
        output[f"evaluation/frames_{i}"] = wandb.Video(frames, fps=10)
    output.pop("evaluation/frames")

    if log_to_wandb:
        wandb.log(output)
    torch.save(model.state_dict(), f"{ckpt_path}/checkpoint_{iter_num}.pt")

if log_to_wandb:
    wandb.finish()
