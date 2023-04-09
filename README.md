# MiniGrid

## Generate dataset

```bash
bash ./scripts/gen-dataset.sh

# or

cd dataset
python gen_dataset.py
python process_data.py
```

## Train

```bash
cd mtdt
python train.py
```

Output checkpoints at `mtdt/ckpts`

## Test

```bash
cd mtdt
python test.py
```

Output gif at `mtdt/gifs`
