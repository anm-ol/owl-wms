<div align="center">

# 🦉 OWL VAEs

<p align="center">
  This is our codebase for VAE training.
</p>

---

</div>

## Basic Information

To get setup just run `pip install -r requirements.txt`.

- Set an **environment variable** for the `WANDB_USER_NAME` to sync correctly w/ Wandb
- To launch training run: `python -m train --config_path /path/to/config.yaml` (or `torchrun`)

## Note for Contributors

This codebase is optimized for remote training on Nvidia GPU clusters while maintaining extensibility and readability. We aim to:

- Keep dependencies minimal to enable quick setup on new instances
- Focus on core CUDA functionality, avoiding extra code for infrequent use-cases
- Avoid over-optimization that doesn't meaningfully improve training throughput
- Maintain only tested, functional code by removing failed experiments
- Replace deprecated architectures when better alternatives are found

In tables below: 🟥 = needs updates, 🟨 = usable but dated, 🟩 = production ready

## Configuration

Configs are YAML files under the `configs/` directory. See existing configs for examples. Config structure is defined in `owl_vaes/configs.py` which specifies model, training and logging parameters.

## Models

Models implement VAE architectures (encoder+decoder+vae). Found in `owl_vaes/models/`. Common building blocks go in `owl_vaes/nn/`. Model implementations should be clean and specific.

| Name | Description | model_id | Status | Example Config |
|------|-------------|----------|---------|----------------|
| DCAE | Basic convolutional AE | dcae | 🟨 | TBD |
| TiToKVAE | Transformer VAE | titok | 🟨 | TBD |
| TiToKVQVAE | VQ version of TiToK | titok_vq | 🟨 | TBD |
| DCVQVAE | VQ version of DCAE | dcae_vq | 🟨 | TBD |
| ProxyTiToKVAE | Proxy version of TiToK | proxy_titok | 🟨 | TBD |
| OobleckVAE | Audio VAE | audio_ae | 🟩 | TBD |
| AudioTransformerDecoder | Transformer audio decoder | tdec | 🟩 | TBD |

## Trainers

Trainers implement specific training approaches. Found in `owl_vaes/trainers/`.

| Name | Description | trainer_id | Status | Example Config |
|------|-------------|------------|---------|----------------|
| RecTrainer | Basic reconstruction | rec | 🟩 | TBD |
| ProxyTrainer | Proxy-based training | proxy | 🟨 | TBD |
| AudioRecTrainer | Audio reconstruction | audio_rec | 🟩 | TBD |
| DecTuneTrainer | Decoder tuning | dec_tune | 🟩 | TBD |
| AudDecTuneTrainer | Audio decoder tuning | audio_dec_tune | 🟩 | TBD |

## Discriminators 

Discriminators for adversarial training. Found in `owl_vaes/discriminators/`.

| Name | Description | model_id | Status | Example Config |
|------|-------------|----------|---------|----------------|
| R3GANDiscriminator | R3 GAN discriminator | r3gan | 🟨 | TBD |
| EncodecDiscriminator | Encodec discriminator | encodec | 🟩 | TBD |

## Data

Data loaders take batch_size and optional kwargs. Found in `owl_vaes/data/`.

| Name | Description | data_id | Status | Example Config |
|------|-------------|---------|---------|----------------|
| MNIST | Basic MNIST | mnist | 🟨 | TBD |
| Local ImageNet | Local 256px ImageNet | local_imagenet_256 | 🟨 | TBD |
| S3 ImageNet | S3-stored ImageNet | s3_imagenet | 🟨 | TBD |
| Local CoD | Local CoD dataset | local_cod | 🟨 | TBD |
| Audio Loader | Generic audio loading | audio_loader | 🟩 | TBD |
| S3 CoD | S3-stored CoD dataset | s3_cod | 🟩 | TBD |
| Local CoD Audio | Local CoD audio | local_cod_audio | 🟩 | TBD |
| S3 CoD Audio | S3-stored CoD audio | s3_cod_audio | 🟩 | TBD |

## Additional Components

- **Losses**: Basic loss functions in `owl_vaes/losses/`
- **Sampling**: Wandb/logging utilities in `owl_vaes/utils/logging.py`
- **Loading**: General utilities in `owl_vaes/utils/__init__.py`
