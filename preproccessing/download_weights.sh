#!/bin/bash

# This script downloads the 'vae' folder from the specified Hugging Face repository.
# It requires the huggingface_hub library to be installed (`pip install huggingface_hub`).

# Define repository and folder details
REPO_ID="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FOLDER_TO_DOWNLOAD="vae"
LOCAL_DIR="checkpoints/Wan2.1"

echo "Downloading folder '${FOLDER_TO_DOWNLOAD}' from repository '${REPO_ID}'..."

# Use huggingface-cli to download the specific folder
huggingface-cli download \
    "$REPO_ID" \
    --repo-type model \
    --include "${FOLDER_TO_DOWNLOAD}/*" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

echo "Download complete. Files are saved in '${LOCAL_DIR}/${FOLDER_TO_DOWNLOAD}'"

huggingface-cli download \
    "Lightricks/LTX-Video" \
    --repo-type model \
    --include "vae/*" \
    --local-dir "checkpoints/LTXV" \
    --local-dir-use-symlinks False