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

# Build, tag, and push with custom tag
./build_and_push.sh v1.0.0
```

The script will:
1. Build the Docker image locally
2. Tag it for your remote registry
3. Push to the configured registry
4. Update `skypilot/config.yaml` with the new image tag

## Multi-Node Training with SkyPilot

### Setup
1. Edit `skypilot/config.yaml` to specify your training configuration:
   ```yaml
   # Change this line to point to your config file
   train.py --config_path configs/YOUR_CONFIG.yml
   ```

2. Optionally adjust the number of nodes and GPU type:
   ```yaml
   resources:
     accelerators: H200:8  # 8 H200s per node
   num_nodes: 2            # Number of nodes
   ```

### Prerequisites
1. Make sure you're authenticated with Google Cloud:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. Make sure you set your Project ID for google cloud.

## Inference

### Tekken Model Inference

Run inference with pre-trained Tekken models to generate video sequences from action inputs.

#### Setup
```bash
# Install additional dependencies for video generation
pip install moviepy

# Ensure VAE and model checkpoints are available
# Update paths in your inference config file
```

#### Configuration
Create or modify an inference configuration file (e.g., `inference/t3_infer.yml`):

```yaml
# Core paths - UPDATE THESE FOR YOUR SETUP
model_config_path: "configs/tekken_nopose_large.yml"
model_ckpt_path: "/path/to/your/model/checkpoint.pt"
actions_npy_path: "/path/to/your/actions.npy"
output_path: "output_video.mp4"

# Inference parameters
starting_frame_index: 0
initial_context_length: 4
num_frames: 120
batch_size: 1
compile: false  # Set to true for faster inference on compatible hardware
```

#### Usage
```bash
# Run inference with default config
python inference/tekken_inference.py

# Run inference with custom config
python inference/tekken_inference.py --config inference/your_config.yml
```

#### Input Requirements
- **Model checkpoint**: Trained Tekken model weights (.pt file)
- **VAE checkpoint**: Compatible VAE decoder for latent-to-pixel conversion
- **Action sequence**: NumPy array (.npy) containing action IDs for generation
- **Initial context**: Dataset sample for conditioning the generation

#### Output
- **Video file**: Generated sequence saved as MP4 (requires moviepy)
- **Fallback**: NumPy array of processed frames if moviepy unavailable
