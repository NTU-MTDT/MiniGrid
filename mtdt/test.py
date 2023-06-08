from main import *
from utils import *
import torch
import os

np.set_printoptions(precision=3)

model_path = "threetasks_ckpt/checkpoint_5.pt"

model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully")

# env_class = getattr(environments, env_name)
# env = env_class(render_mode="rgb_array")

if not os.path.exists("gifs"):
    os.mkdir("gifs")

target_return = np.array([-100, 0, 0])

actual_return = []
return_error = []
frames = []
for k in tqdm(range(100)):
    seed = random.randint(0, 100000)
    env.reset(seed=seed)
    render = len(frames) < 10

    result = eval(
        model=model,
        env=env,
        rw=target_return,
        seed=seed,
        temperature=1.0,
        render=render,
    )
    actual_return.append(result["actual_return"])
    return_error.append(abs(target_return - actual_return[-1]))

    if render:
        frame = result["frames"]
        frame = np.concatenate([frame] + [frame[-1:] for _ in range(5)], axis=0)
        frames.append(frame)

actual_return = np.mean(actual_return, axis=0)
return_error_mean = np.mean(return_error, axis=0)
return_error_std = np.std(return_error, axis=0)
frames = np.concatenate(frames, axis=0)
print(frames.shape)

print(f"Seed: {seed}")
print(f"Episode length: {result['episode_length']}")
print(f"Target return:\t{target_return}")
print(f"Actual return:\t{actual_return}")
print(f"Return error mean:\t{return_error_mean}")
print(f"Return error std:\t{return_error_std}")
print(f"Action sequence: {result['actions']}")

# save the result as gif
print("Rendering GIF")
display_frames_as_gif(frames, name=f"gifs/{target_return}_{actual_return}.gif")

# render_gif(
#     env,
#     seed,
#     result["actions"],
#     name=f"gifs/seed_{seed}_{target_return}_{actual_return}.gif",
# )
