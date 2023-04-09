from main import *
from utils import *
import torch
import seaborn as sns

np.set_printoptions(precision=3)

model_path = "ckpts/checkpoint_49.pt"

model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully")
env = EmptyEnv(render_mode="rgb_array")

if not os.path.exists("gifs"):
    os.mkdir("gifs")

target_return = np.array([0.8, 0, 10])

actual_return = []
return_error = []
for k in tqdm(range(5)):
    seed = random.randint(0, 100000)
    env.reset(seed=seed)
    result = eval(
        model=model, env=env, rw=target_return, seed=seed, temperature=1.0
    )
    actual_return.append(result["actual_return"])
    return_error.append(abs(target_return - actual_return[-1]))
actual_return = np.mean(actual_return, axis=0)
return_error_mean = np.mean(return_error, axis=0)
return_error_std = np.std(return_error, axis=0)

print(f"Seed: {seed}")
print(f"Episode length: {result['episode_length']}")
print(f"Target return:\t{target_return}")
print(f"Actual return:\t{actual_return}")
print(f"Return error mean:\t{return_error_mean}")
print(f"Return error std:\t{return_error_std}")
print(f"Action sequence: {result['actions']}")

# save the result as gif
print("Rendering GIF")
render_gif(
    env,
    seed,
    result["actions"],
    name=f"gifs/seed_{seed}_{target_return}_{actual_return}.gif",
)
