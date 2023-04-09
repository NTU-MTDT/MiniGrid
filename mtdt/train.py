from main import *
import os
import wandb
from torch.nn import CrossEntropyLoss
from decision_transformer.training.seq_trainer import SequenceTrainer


wandb.init(entity="mtdt", project="MTDT", name=run_name, config=config)
outputs = []

print(f"max_iterations {max_iterations}")
print(f"num_steps_per_iter {num_steps_per_iter}")

if not os.path.exists("ckpts"):
    os.mkdir("ckpts")
    
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


for i in range(max_iterations):
    output = trainer.train_iteration(
        num_steps=num_steps_per_iter, iter_num=i, print_logs=True
    )
    outputs.append(output)

    log_output = output.copy()
    for k, v in output.items():
        if type(v) == list:
            for j in range(len(v)):
                log_output[f"{k}_{j}"] = v[j]
            log_output.pop(k)
        else:
            log_output[f"{k}"] = v
    wandb.log(log_output)
    torch.save(model.state_dict(), f"ckpts/checkpoint_{i}.pt")

    # if outputs[-1]['evaluation/exact_match'] > best_exactmatch:
    #     torch.save(model.state_dict(), 'best_model.pt')
    #     best_exactmatch = outputs[-1]['evaluation/exact_match']
wandb.finish()

# %%
# errors = np.array([output["evaluation/error_mean"] for output in outputs])
# stds = np.array([output["evaluation/error_std"] for output in outputs])
# confidences = np.array([output["evaluation/error_confidence_interval"] for output in outputs])

# with open("train_logs.pkl", "wb") as f:
#     pickle.dump((errors, stds, confidences), f)
