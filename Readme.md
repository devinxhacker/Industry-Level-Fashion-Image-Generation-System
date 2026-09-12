# Fashion Image Generation with VAE and DCGAN

This repository completes the assignment with two PyTorch systems trained on Fashion-MNIST:

1. A convolutional VAE that learns a probabilistic clothing representation, reconstructs images, samples new designs, and demonstrates smooth latent interpolation.
2. A DCGAN that learns to generate synthetic clothing images from random noise.

## Project layout

```text
.
├── src/models.py       # VAE, generator, discriminator, and VAE loss
├── src/data.py         # Fashion-MNIST loaders and image-grid export
├── train.py            # CLI for smoke tests and training
├── report.md           # Assignment write-up and results template
├── SMOKE_TEST.md       # Verification checklist
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first training run downloads Fashion-MNIST into `data/`. Generated checkpoints, metrics, and figures are written to `outputs/`.

## Run

Run the architecture and tensor-shape check without downloading data:

```bash
python train.py --smoke-test
```

Train the VAE and create reconstructions, random samples, and a latent interpolation grid:

```bash
python train.py --model vae --epochs 10
```

Train the DCGAN and create a generated sample grid:

```bash
python train.py --model gan --epochs 20
```

For a quick experiment, reduce the dataset and epoch count:

```bash
python train.py --model both --limit 4096 --epochs 2
```

Use `--device mps` on a compatible Apple Silicon machine, `--device cuda` on an NVIDIA machine, or leave the default `--device auto`.

## Lite UI

Launch the local interface after installing the requirements:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit. The app provides a synthetic collection generator, VAE reconstruction and latent interpolation from uploaded images, and training metrics with saved sample previews. It automatically uses CUDA, Apple MPS, or CPU and loads checkpoints from `outputs/`.

## Design choices

The VAE uses `[0, 1]` inputs, a sigmoid decoder, binary cross-entropy reconstruction loss, and KL divergence to a standard normal prior. The GAN uses `[-1, 1]` inputs, a tanh generator, and `BCEWithLogitsLoss` for stable discriminator training. The interpolation figure encodes two examples with the VAE mean vectors and decodes eight linear points between them.

See [report.md](report.md) for the methodology, equations, reproducibility commands, results table, limitations, and recommended DeepFashion next steps.
