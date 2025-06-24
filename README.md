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

In tables below: 🟥 = needs updates, 🟨 = usable but dated, 🟩 = good and up to date

## Configuration

Configs are YAML files under the `configs/` directory. See existing configs for examples. Config structure is defined in `owl_vaes/configs.py` which specifies model, training and logging parameters.

## Models

Models implement VAE architectures (encoder+decoder+vae). Found in `owl_vaes/models/`. Common building blocks go in `owl_vaes/nn/`. Model implementations should be clean and specific.

| Name | Description | model_id | Status | Example Config |
|------|-------------|----------|---------|----------------|
| DCAE | Basic convolutional AE | dcae | 🟩 | configs/cod_128x_depth.yml |
| TiToKVAE | Transformer VAE | titok | 🟩 | configs/titok.yml |
| TiToKVQVAE | VQ version of TiToK | titok_vq | 🟨 | configs/titok_vq.yml |
| DCVQVAE | VQ version of DCAE | dcae_vq | 🟨 | TBD |
| ProxyTiToKVAE | Proxy version of TiToK | proxy_titok | 🟨 | TBD |
| OobleckVAE | Audio VAE | audio_ae | 🟩 | configs/audio_ae.yml |
| AudioTransformerDecoder | Transformer audio decoder | tdec | 🟩 | TBD |

## Trainers

Trainers implement specific training approaches. Found in `owl_vaes/trainers/`.

| Name | Description | trainer_id | Status | Example Config |
|------|-------------|------------|---------|----------------|
| RecTrainer | Basic reconstruction-only for images | rec | 🟩 | configs/cod_128x_depth.yml |
| ProxyTrainer | Proxy-based training | proxy | 🟨 | TBD |
| AudioRecTrainer | Audio reconstruction-only | audio_rec | 🟩 | configs/audio_ae.yml |
| DecTuneTrainer | Adversarial Decoder post-training | dec_tune | 🟨 | configs/simple_dec_tune.yml |
| AudDecTuneTrainer | Adversarial Decoder post-training for audio | audio_dec_tune | 🟩 | configs/audio_ae_tune.yml |

## Discriminators 

Discriminators for adversarial training. Found in `owl_vaes/discriminators/`.

| Name | Description | model_id | Status | Example Config |
|------|-------------|----------|---------|----------------|
| R3GANDiscriminator | R3GAN discriminator | r3gan | 🟩 | configs/simple_dec_tune.yml |
| EncodecDiscriminator | Encodec discriminator | encodec | 🟩 | configs/audio_ae_tune.yml |

## Data

Data loaders take batch_size and optional kwargs. Found in `owl_vaes/data/`.

| Name | Description | data_id | Status | Example Config |
|------|-------------|---------|---------|----------------|
| MNIST | Just MNIST | mnist | 🟩 | TBD |
| Local ImageNet | Local 256px ImageNet | local_imagenet_256 | 🟨 | TBD |
| S3 ImageNet | S3-stored ImageNet | s3_imagenet | 🟨 | TBD |
| Local CoD | Local CoD dataset | local_cod | 🟨 | TBD |
| Audio Loader | Generic audio loading | audio_loader | 🟨 | TBD |
| S3 CoD | S3-stored CoD frame dataset | s3_cod | 🟩 | configs/cod_128x_huge_kl.yml |
| Local CoD Audio | Local CoD audio (directory of wavs) | local_cod_audio | 🟩 | TBD |
| S3 CoD Audio | S3-stored CoD waveform audio | s3_cod_audio | 🟩 | configs/audio_ae.yml |
| S3 CoD Features | S3-stored CoD with depth+flow | s3_cod_features | 🟩 | configs/cod_128x_depth.yml | 

## Additional Components

- **Losses**: Basic loss functions in `owl_vaes/losses/`
- **Sampling**: Wandb/logging utilities in `owl_vaes/utils/logging.py`
- **Loading**: General utilities in `owl_vaes/utils/__init__.py`
