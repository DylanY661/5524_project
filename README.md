# Setup Instructions

First install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

Make sure the Python version used is `>= 3.10`.

It is also recommended to run the script on an OSC cluster.

## Required Files

The embedding and checkpoint files are too large for GitHub and are hosted on Google Drive. Download and extract them to the directory before running the script.

### Embeddings

**Download:** [embeddings.zip](https://drive.google.com/file/d/1N6_GM4XZdtJy2kahWSkqFnWX7eFzG6Q1/view?usp=sharing)

After downloading, extract the contents:
```bash
unzip embeddings.zip
```

The following files should be present after extraction:

| Dataset | File |
|---------|------|
| ImageNet-1K | `imagenet_clip_embeddings.pt` |
| ImageNet-1K | `imagenet_dino_embeddings.pt` |
| ImageNet-R | `imagenet_r_clip_embeddings.pt` |
| ImageNet-R | `imagenet_r_dino_embeddings.pt` |
| CIFAR-100 | `cifar100_clip_embeddings.pt` |
| CIFAR-100 | `cifar100_dino_embeddings.pt` |

### Checkpoints

**Download:** [checkpoints.zip](https://drive.google.com/file/d/1vphGjJiUIrcMLtx2TUnhD4w9mDxBQb3g/view?usp=sharing)

After downloading, extract the contents:
```bash
unzip checkpoints.zip
```

The following files should be present after extraction:

| Model | Checkpoint Path |
|-------|-----------------|
| Model 1 (MLP Fusion) | `checkpoints/mlp/best_model.pt` |
| Model 1 (MLP Fusion) | `checkpoints/mlp/latest_checkpoint.pt` |
| Model 2 (Joint Fusion) | `checkpoints/joint/best_model.pt` |
| Model 2 (Joint Fusion) | `checkpoints/joint/latest_checkpoint.pt` |

---

# Running Instructions

## Option 1: OSC (Preferred)

If running on OSC, simply run the command:
```bash
sbatch ./main.sbatch
```

This will automatically submit a batch job to OSC to run the evaluation script.

The output will be directed to a file named `results_{jobid}.txt`, where `jobid` is the batch job ID assigned by OSC.

## Option 2: Direct Execution

The script can also be run directly inside the terminal.

Make sure `python >= 3.10.0` is being used, then run either:
```bash
python main.py
```

or
```bash
python3 main.py
```

depending on your OS.

Results should print to terminal.

---

Expected format and results are given in epxected_results.txt. The output fo the evaluation script should match.

# Additional Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--model` | `-m` | string | `all` | Model to evaluate. Choices: `advanced_1`, `advanced_2`, `all` |
| `--embeddings_dir` | `-e` | string | `embeddings` | Directory containing precomputed embeddings |
| `--checkpoint_dir` | `-c` | string | `.` | Base directory for model checkpoints |
| `--dataset` | `-d` | string | `all` | Dataset to evaluate on. Choices: `imagenet`, `imagenet-r`, `cifar100`, `all` |

## Example Usage

Evaluate Model 1 (MLP Fusion) on all datasets:
```bash
python main.py --model advanced_1
```

Evaluate Model 2 (Joint Fusion) on ImageNet-1K only:
```bash
python main.py --model advanced_2 --dataset imagenet
```

Evaluate all models with custom directories:
```bash
python main.py --model all --embeddings_dir /path/to/embeddings --checkpoint_dir /path/to/checkpoints
```
