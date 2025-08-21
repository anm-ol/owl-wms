#!/bin/bash

pip install -r requirements.txt
cd preproccessing
huggingface-cli download Summer-193/t3_data_V2 --repo-type dataset --local-dir .

# This script downloads the 'vae' folder from the specified Hugging Face repository.
# It requires the huggingface_hub library to be installed (`pip install huggingface_hub`).

# Define repository and folder details
REPO_ID="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FOLDER_TO_DOWNLOAD="vae"
LOCAL_DIR="checkpoints/Wan2.1"
# Use huggingface-cli to download the specific folder
huggingface-cli download \
    "$REPO_ID" \
    --repo-type model \
    --include "${FOLDER_TO_DOWNLOAD}/*" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

huggingface-cli download \
    "Lightricks/LTX-Video" \
    --repo-type model \
    --include "vae/*" \
    --local-dir "checkpoints/LTXV" \
    --local-dir-use-symlinks False

cd .. 
python preproccessing/prepare_data_v2.py

