"""Neural network architectures used by the assignment."""

import torch
from torch import nn


class VAE(nn.Module):
    """Convolutional VAE for 28x28 grayscale clothing images."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z).view(-1, 64, 7, 7)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class Generator(nn.Module):
    """DCGAN generator mapping a 100-dimensional noise vector to 28x28 images."""

    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.main(noise)


class Discriminator(nn.Module):
    """DCGAN discriminator returning one probability per 28x28 image."""

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.main(images).flatten()


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction BCE, and KL terms averaged per image."""
    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstruction, target, reduction="sum"
    ) / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / target.size(0)
    return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss
