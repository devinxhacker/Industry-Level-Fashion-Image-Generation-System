"""Train the Fashion-MNIST VAE and DCGAN, or run a fast architecture smoke test.

Examples:
    python train.py --smoke-test
    python train.py --model vae --epochs 10
    python train.py --model gan --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.data import fashion_loader, save_grid, synthetic_loader
from src.models import Discriminator, Generator, VAE, vae_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def write_metrics(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_vae(args: argparse.Namespace, device: torch.device) -> None:
    output = Path(args.output) / "vae"
    loader = fashion_loader(args.data, args.batch_size, limit=args.limit)
    model = VAE(args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(images)
            loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, images, mu, logvar)
            loss.backward()
            optimizer.step()
            totals["loss"] += loss.item()
            totals["reconstruction"] += reconstruction_loss.item()
            totals["kl"] += kl_loss.item()
        count = len(loader)
        row = {"epoch": float(epoch), **{key: value / count for key, value in totals.items()}}
        history.append(row)
        print(f"VAE epoch {epoch:03d}: loss={row['loss']:.3f} recon={row['reconstruction']:.3f} kl={row['kl']:.3f}")

    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "latent_dim": args.latent_dim}, output / "vae.pt")
    write_metrics(history, output / "metrics.csv")
    model.eval()
    with torch.no_grad():
        random_samples = model.decode(torch.randn(64, args.latent_dim, device=device))
        images, _ = next(iter(loader))
        images = images[:2].to(device)
        mu, _ = model.encode(images)
        interpolation = torch.cat(
            [model.decode(torch.lerp(mu[0], mu[1], step / 7)) for step in range(8)]
        )
        reconstructions, _, _ = model(images)
    save_grid(random_samples, output / "random_samples.png")
    save_grid(torch.cat([images.cpu(), reconstructions.cpu()]), output / "reconstructions.png")
    save_grid(interpolation, output / "latent_interpolation.png")


def train_gan(args: argparse.Namespace, device: torch.device) -> None:
    output = Path(args.output) / "gan"
    loader = fashion_loader(args.data, args.batch_size, limit=args.limit, gan=True)
    generator = Generator(args.noise_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, args.noise_dim, 1, 1, device=device)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        totals = {"discriminator": 0.0, "generator": 0.0}
        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real_images), real_targets)
            fake_images = generator(torch.randn(batch_size, args.noise_dim, 1, 1, device=device))
            fake_loss = criterion(discriminator(fake_images.detach()), fake_targets)
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss = criterion(discriminator(fake_images), real_targets)
            generator_loss.backward()
            generator_optimizer.step()
            totals["discriminator"] += discriminator_loss.item()
            totals["generator"] += generator_loss.item()
        count = len(loader)
        row = {"epoch": float(epoch), **{key: value / count for key, value in totals.items()}}
        history.append(row)
        print(f"GAN epoch {epoch:03d}: d_loss={row['discriminator']:.3f} g_loss={row['generator']:.3f}")

    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "noise_dim": args.noise_dim},
        output / "dcgan.pt",
    )
    write_metrics(history, output / "metrics.csv")
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise)
    save_grid(samples, output / "samples.png", value_range=(-1, 1))


def smoke_test(device: torch.device) -> None:
    loader = synthetic_loader(batch_size=4, batches=1)
    images, _ = next(iter(loader))
    vae = VAE(8).to(device)
    reconstruction, mu, logvar = vae(images.to(device))
    total, reconstruction_loss, kl_loss = vae_loss(reconstruction, images.to(device), mu, logvar)
    generator = Generator(16).to(device)
    discriminator = Discriminator().to(device)
    generated = generator(torch.randn(4, 16, 1, 1, device=device))
    scores = discriminator(generated)
    assert reconstruction.shape == images.shape
    assert total.isfinite() and reconstruction_loss.isfinite() and kl_loss.isfinite()
    assert generated.shape == images.shape and scores.shape == (4,)
    print(f"smoke test passed on {device}: VAE {tuple(reconstruction.shape)}, DCGAN {tuple(generated.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vae", "gan", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset size for quick experiments")
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = device_for(args.device)
    if args.smoke_test:
        smoke_test(device)
        return
    if args.model in ("vae", "both"):
        train_vae(args, device)
    if args.model in ("gan", "both"):
        train_gan(args, device)


if __name__ == "__main__":
    main()
