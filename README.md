# Tekken World Model

This repository is a fork of owl-wms, heavily modified to develop, train, and evaluate action-conditioned world models specifically for Tekken 8 gameplay. The original codebase has been extended to support multiple VAE backbones, custom data loaders for Tekken, specialized trainers, and an inference pipeline focused on generating Tekken gameplay.

## Project Overview

The primary goal of this project is to create a transformer-based world model capable of predicting future video frames of Tekken gameplay given a sequence of past frames and player actions.

The architecture consists of two main components:

- **A VAE (owl-vaes)**: A Variational Autoencoder is used as a pre-processing step to compress high-resolution game frames into a lower-dimensional latent representation. This fork supports multiple VAEs, including LTX, Wan, and custom-trained DCAE models.
- **The World Model (owl-wms)**: A DiT (Diffusion Transformer) backbone learns to predict the next sequence of latent vectors conditioned on previous latents and a sequence of player actions.

## 1. Getting Started

Follow these steps to set up the repository for local development and training.

### Initial Setup

**Clone the Repository**: Clone the repository and its submodules (which includes owl-vaes).
```bash
git clone --recursive -j8 <your-repo-url>
cd owl-wms
```

**Install Dependencies**: Install the required Python packages for both the main project and the VAE submodule.
```bash
pip install -r requirements.txt
pip install -r owl-vaes/requirements.txt
```

**Configure Environment**: Copy the environment file template and add your specific credentials, particularly your Weights & Biases API key.
```bash
cp .env.example .env
```
Edit the .env file to add your `WANDB_API_KEY` and any necessary S3/Tigris credentials if you are using cloud storage for datasets.

**Login to WandB**: Authenticate with Weights & Biases for experiment tracking.
```bash
wandb login
```

## 2. Data Preparation Workflow

The world model is trained on latent vectors, not raw video frames. You must first process your raw Tekken gameplay footage into these latents using the provided VAEs.

### Step 2.1: Raw Video to NPZ

This step is currently a manual prerequisite. You need to convert your raw gameplay videos into `.npz` files containing arrays for `images`, `actions_p1`, `states`, and `valid_frames`. The expected directory structure for this raw data is:

```
preproccessing/data_v3/
├── train/
│   ├── round_001.npz
│   └── round_002.npz
└── val/
    ├── round_101.npz
    └── round_102.npz
```

### Step 2.2: NPZ to VAE Latents (Caching)

Once you have the raw `.npz` files, use the provided scripts to encode the image data into latent vectors using a chosen VAE. This is a crucial step that "caches" the data for efficient training.

**Download Pre-trained VAEs**:
Run the script to download the supported community VAEs (Wan and LTX).
```bash
bash preproccessing/download_weights.sh
```
Custom-trained VAEs are already included in `preproccessing/checkpoints/`.

**Run the Caching Script**: The `prepare_latents_owl.py` script handles the encoding process and is optimized for multi-GPU usage.

Example Command (using a custom DCAE VAE with pose data):
```bash
python preproccessing/prepare_latents_owl.py \
    --vae-ckpt-dir "preproccessing/checkpoints/tekken_vae_H200_v6" \
    --data-dir "preproccessing/data_v3" \
    --pose-dir "preproccessing/t3_pose" \
    --output-dir "preproccessing/cached_dcae" \
    --batch-size 16
```

This script reads `.npz` files from `--data-dir`, uses the specified VAE to encode images (optionally merging pose data from `--pose-dir`), and saves the resulting latents, actions, and states into the `--output-dir`.

## 3. Training the World Model

The model can be trained locally on a single machine (with multiple GPUs) or on a multi-node cluster using SkyPilot.

### Local / Single-Node Training

Use `torchrun` to launch a multi-GPU training job on a single machine.

```bash
# Example for training on 2 GPUs using the LTX VAE latents
torchrun --nproc_per_node=2 train.py --config_path configs/tekken_action_ltx.yml
```

### Multi-Node Training with SkyPilot

For large-scale training, SkyPilot is used to manage multi-node GPU clusters.

**Setup SkyPilot**: Install and authenticate with your cluster.
```bash
# Install SkyPilot
pip install -U skypilot

# Authenticate with your cluster endpoint
sky api login -e https://owlskypilot:<password>@cluster.openworldlabs.ai
```

**Build and Push Docker Image**: The `build_and_push.sh` script containerizes the current codebase and pushes it to the Docker registry configured in your `.env` file.
```bash
# This script builds, tags, pushes, and updates skypilot/config.yaml
./build_and_push.sh
```

**Configure and Launch**: Edit `skypilot/config.yaml` to point to your desired training configuration file, then launch the job.
```bash
# In skypilot/config.yaml, update the train.py command:
...
run: |
  ...
  torchrun \
    ...
    train.py --config_path configs/YOUR_TEKKEN_CONFIG.yml --nccl_timeout 1000

# Launch a 2-node job with 8 H200 GPUs each
export EXPERIMENT_NAME=tekken-v3-large-model
sky launch --infra kubernetes --gpus H200:8 --num-nodes 2 --name $EXPERIMENT_NAME skypilot/config.yaml
```

## 4. Inference

**⚠️ Note**: The inference pipeline is currently experimental and will undergo significant changes soon.

To generate a video from a trained model checkpoint, use the `tekken_inference.py` script. This requires a configuration file, a model checkpoint, and a sequence of actions.

**Example Usage**:
```bash
python inference/tekken_inference.py \
    --config_path configs/tekken_nopose_large.yml \
    --model_ckpt_path /path/to/your/model_checkpoint.pt \
    --actions_npy_path /path/to/your/action_sequence.npy \
    --output_path generated_video.mp4 \
    --num_frames 180 \
    --compile
```

## 5. Project Status & Key Components

This project has evolved significantly from the original owl-wms repo. Here is a guide to the current status of key components.

**⚠️ Important Training Notes**:
- **WM DiT training works great** with custom-trained 2D VAEs on Tekken data
- **LTX training works okay but not consistent** - loses coherence very quickly during generation
- **Wan pipeline is currently garbage** and needs lots of debugging
- **Best performing configurations** are `tekken_nopose_large.yml` and `tekken_pose_v3_L.yml` which show great results

### Supported VAEs

| VAE Name | Checkpoint Location | Description |
|----------|-------------------|-------------|
| LTX-Video | `preproccessing/checkpoints/LTXV/vae` | A high-quality VAE from Lightricks. Works okay but loses coherence quickly. |
| Wan 2.1 | `preproccessing/checkpoints/Wan2.1/vae` | A VAE from Wan-AI. **Pipeline currently broken** and needs debugging. |
| Custom DCAE (Pose) | `preproccessing/checkpoints/tekken_vae_H200_v6` | **Recommended**: Custom VAE trained on Tekken data with pose. Works great with WM DiT training. |
| Custom DCAE (No Pose) | `preproccessing/checkpoints/t3_VAE_nopose_v1` | **Recommended**: Custom VAE trained on Tekken data without pose. Works great with WM DiT training. |

### Training Configurations by Status

#### 🟢 **Best Performing / Recommended**
| Configuration File | VAE Used | Description |
|------------------|----------|-------------|
| `tekken_nopose_large.yml` | Custom No-Pose VAE | **Best results**: Large model trained without pose data. WM DiT training works great. |
| `tekken_pose_v3_L.yml` | Custom DCAE (Pose) | **Best results**: Large model (d_model: 2048) with pose data. WM DiT training works great. |
| `tekken_dcae_v6.yml` | Custom DCAE (Pose) | **Recommended**: Works great with custom VAE and WM DiT training |
| `tekken_nopose.yml` | Custom No-Pose VAE | **Recommended**: Standard size model, works great with custom VAE |

#### 🟡 **Working but with Issues**
| Configuration File | VAE Used | Description |
|------------------|----------|-------------|
| `tekken_action_ltx.yml` | LTX-Video | Works okay but not consistent, loses coherence very quickly |

#### 🔴 **Known Issues / Broken**
| Configuration File | VAE Used | Description |
|------------------|----------|-------------|
| `tekken_action_wan.yml` | Wan 2.1 | **Pipeline currently garbage**, needs lots of debugging |

## 6. Current Development Priorities

1. **Debug and fix Wan pipeline** - currently broken and needs extensive debugging
2. **Improve LTX coherence issues** - address quick loss of coherence during generation
3. **Document tekken_rft_v2** capabilities and usage
4. **Stabilize inference pipeline** (currently experimental)
5. **Optimize tekken_action_caching sampler** further based on its current success
6. **Continue optimizing custom VAE + WM DiT training** which is currently working great
## Inference

### Tekken Model Inference

Run inference with pre-trained Tekken models to generate video sequences from action inputs.

#### Setup
```bash
# In skypilot/config.yaml, update the train.py command:
...
run: |
  ...
  torchrun \
    ...
    train.py --config_path configs/YOUR_TEKKEN_CONFIG.yml --nccl_timeout 1000

# Launch a 2-node job with 8 H200 GPUs each
export EXPERIMENT_NAME=tekken-v3-large-model
sky launch --infra kubernetes --gpus H200:8 --num-nodes 2 --name $EXPERIMENT_NAME skypilot/config.yaml
```

## 4. Inference

**⚠️ Note**: The inference pipeline is currently experimental and will undergo significant changes soon.

To generate a video from a trained model checkpoint, use the `tekken_inference.py` script. This requires a configuration file, a model checkpoint, and a sequence of actions.

**Example Usage**:
```bash
python inference/tekken_inference.py \
    --config_path configs/tekken_nopose_large.yml \
    --model_ckpt_path /path/to/your/model_checkpoint.pt \
    --actions_npy_path /path/to/your/action_sequence.npy \
    --output_path generated_video.mp4 \
    --num_frames 180 \
    --compile
```

## 5. Project Status & Key Components

This project has evolved significantly from the original owl-wms repo. Here is a guide to the current status of key components.

**⚠️ Important Training Notes**:
- **WM DiT training works great** with custom-trained 2D VAEs on Tekken data
- **LTX training works okay but not consistent** - loses coherence very quickly during generation
- **Wan pipeline is currently garbage** and needs lots of debugging
- **Best performing configurations** are `tekken_nopose_large.yml` and `tekken_pose_v3_L.yml` which show great results

### Supported VAEs

| VAE Name | Checkpoint Location | Description |
|----------|-------------------|-------------|
| LTX-Video | `preproccessing/checkpoints/LTXV/vae` | A high-quality VAE from Lightricks. Works okay but loses coherence quickly. |
| Wan 2.1 | `preproccessing/checkpoints/Wan2.1/vae` | A VAE from Wan-AI. **Pipeline currently broken** and needs debugging. |
| Custom DCAE (Pose) | `preproccessing/checkpoints/tekken_vae_H200_v6` | **Recommended**: Custom VAE trained on Tekken data with pose. Works great with WM DiT training. |
| Custom DCAE (No Pose) | `preproccessing/checkpoints/t3_VAE_nopose_v1` | **Recommended**: Custom VAE trained on Tekken data without pose. Works great with WM DiT training. |

### Training Configurations by Status

#### 🟢 **Best Performing / Recommended**
| Configuration File | VAE Used | Description |
|------------------|----------|-------------|
| `tekken_nopose_large.yml` | Custom No-Pose VAE | **Best results**: Large model trained without pose data. WM DiT training works great. |
| `tekken_pose_v3_L.yml` | Custom DCAE (Pose) | **Best results**: Large model (d_model: 2048) with pose data. WM DiT training works great. |
| `tekken_dcae_v6.yml` | Custom DCAE (Pose) | **Recommended**: Works great with custom VAE and WM DiT training |
| `tekken_nopose.yml` | Custom No-Pose VAE | **Recommended**: Standard size model, works great with custom VAE |

#### 🟡 **Working but with Issues**
| Configuration File | VAE Used | Description |
|------------------|----------|-------------|
| `tekken_action_ltx.yml` | LTX-Video | Works okay but not consistent, loses coherence very quickly |

#### 🔴 **Known Issues / Broken**
| Configuration File | VAE Used | Description |
|------------------|----------|-------------|
| `tekken_action_wan.yml` | Wan 2.1 | **Pipeline currently garbage**, needs lots of debugging |

## 6. Current Development Priorities

1. **Debug and fix Wan pipeline** - currently broken and needs extensive debugging
2. **Improve LTX coherence issues** - address quick loss of coherence during generation
3. **Document tekken_rft_v2** capabilities and usage
4. **Stabilize inference pipeline** (currently experimental)
5. **Optimize tekken_action_caching sampler** further based on its current success
6. **Continue optimizing custom VAE + WM DiT training** which is currently working great