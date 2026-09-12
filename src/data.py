"""Dataset and visualization helpers."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def fashion_loader(
    root: str = "data",
    batch_size: int = 128,
    train: bool = True,
    limit: int | None = None,
    gan: bool = False,
    download: bool = True,
) -> DataLoader:
    """Create a Fashion-MNIST loader with the model-appropriate normalization."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if gan
        else [transforms.ToTensor()]
    )
    dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def synthetic_loader(batch_size: int = 8, batches: int = 2) -> DataLoader:
    """Return deterministic-shaped random data for dependency-free model smoke tests."""
    images = torch.rand(batch_size * batches, 1, 28, 28)
    labels = torch.zeros(len(images), dtype=torch.long)
    return DataLoader(list(zip(images, labels)), batch_size=batch_size)


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, value_range: tuple[float, float] = (0, 1)) -> None:
    """Save a tensor batch as a simple image grid."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().clamp(*value_range)
    images = (images - value_range[0]) / (value_range[1] - value_range[0])
    columns = min(nrow, len(images))
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.5), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
    figure.tight_layout(pad=0.1)
    figure.savefig(path, dpi=160)
    plt.close(figure)
