"""
Data pipeline for CIFAR-10.

Handles downloading, normalization, and DataLoader creation.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data", num_workers: int = 0):
    """Create train and test DataLoaders for CIFAR-10.

    Normalization uses the standard CIFAR-10 channel means and stds.

    Args:
        batch_size: Batch size for both loaders.
        data_dir: Directory to download/cache CIFAR-10.
        num_workers: Number of data loading workers.

    Returns:
        (train_loader, test_loader) tuple.
    """
    # Standard CIFAR-10 normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
