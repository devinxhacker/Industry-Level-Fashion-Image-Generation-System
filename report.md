# Industry-Level Fashion Image Generation System

## Abstract

This project implements two complementary generative models for Fashion-MNIST clothing images: a convolutional Variational Autoencoder (VAE) for representation learning and smooth latent-space interpolation, and a Deep Convolutional GAN (DCGAN) for synthetic image generation.

## Dataset and preprocessing

Fashion-MNIST contains 70,000 grayscale clothing images at 28 x 28 pixels across 10 classes. The VAE receives tensors in `[0, 1]` because its decoder ends in a sigmoid. The GAN receives tensors in `[-1, 1]` because its generator ends in `tanh`.

## Methodology

### VAE

The encoder uses two stride-2 convolutional blocks and predicts a mean $\\mu$ and log-variance $\\log\\sigma^2$. The reparameterization trick samples $z = \\mu + \\sigma\\odot\\epsilon$, where $\\epsilon \\sim \\mathcal{N}(0, I)$. The loss is:

$$L_{VAE} = BCE(x, \\hat{x}) + D_{KL}(q(z|x)||\\mathcal{N}(0,I))$$

The saved `latent_interpolation.png` demonstrates representation learning by decoding linear points between two encoded clothing examples.

### DCGAN

The generator maps a 100-dimensional Gaussian noise vector to a 28 x 28 image. The discriminator classifies real and generated images. `BCEWithLogitsLoss` is used for numerical stability, with the discriminator trained on real labels of 1 and fake labels of 0, followed by a generator update using target label 1.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --smoke-test
python train.py --model vae --epochs 10
python train.py --model gan --epochs 20
```

Training writes checkpoints, CSV loss histories, and PNG figures under `outputs/vae` and `outputs/gan`. For a quick local run, add `--limit 4096 --epochs 2`.

## Results template

After training, insert the generated figures and report:

| Model | Reconstruction / adversarial result | Evidence |
| --- | --- | --- |
| VAE | Report final BCE, KL, and total loss from `outputs/vae/metrics.csv`. Discuss whether interpolation changes smoothly between clothing structures. | `random_samples.png`, `reconstructions.png`, `latent_interpolation.png` |
| DCGAN | Report final discriminator and generator losses from `outputs/gan/metrics.csv`. Discuss sample diversity, realism, and any mode collapse. | `samples.png` |

## Limitations and next steps

Fashion-MNIST is a 28 x 28 prototype dataset, not a production fashion catalog. Scaling to DeepFashion would require higher-resolution preprocessing, more GPU memory, conditional labels or text prompts, stronger evaluation such as FID/KID, and dataset licensing/privacy review. GAN loss values alone do not prove visual quality, so generated samples must be inspected alongside a quantitative metric.
