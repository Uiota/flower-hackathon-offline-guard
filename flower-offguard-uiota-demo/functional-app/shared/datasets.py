"""
Dataset handling and non-IID partitioning for federated learning
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing purposes."""

    def __init__(self, size: int = 1000, input_dim: int = 784, num_classes: int = 10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate synthetic data
        np.random.seed(42)
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float = 0.5) -> List[np.ndarray]:
    """
    Partition data indices using Dirichlet distribution for non-IID simulation.

    Args:
        labels: Dataset labels
        num_clients: Number of clients to partition data among
        alpha: Dirichlet concentration parameter (lower = more non-IID)

    Returns:
        List of index arrays for each client
    """
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for class_idx, class_data_indices in enumerate(class_indices):
        # Shuffle class indices
        np.random.shuffle(class_data_indices)

        # Split according to Dirichlet distribution
        proportions = label_distribution[class_idx]
        proportions = proportions / proportions.sum()  # Normalize

        # Calculate split points
        split_points = (np.cumsum(proportions) * len(class_data_indices)).astype(int)
        split_points[-1] = len(class_data_indices)  # Ensure all data is used

        # Distribute indices to clients
        start_idx = 0
        for client_idx, end_idx in enumerate(split_points):
            if end_idx > start_idx:
                client_indices[client_idx].extend(class_data_indices[start_idx:end_idx])
            start_idx = end_idx

    # Convert to numpy arrays and shuffle each client's data
    for client_idx in range(num_clients):
        client_indices[client_idx] = np.array(client_indices[client_idx])
        np.random.shuffle(client_indices[client_idx])

    return client_indices


def get_mnist_client_data(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    alpha: float = 0.5,
    data_path: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST data partition for a specific client.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        batch_size: Batch size for data loaders
        alpha: Non-IID parameter for Dirichlet distribution
        data_path: Path to store dataset

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Ensure data directory exists
    Path(data_path).mkdir(parents=True, exist_ok=True)

    # Load full datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    # Set deterministic seed for consistent partitioning
    np.random.seed(42)

    # Partition training data using Dirichlet distribution
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    client_indices = dirichlet_partition(train_labels, num_clients, alpha)

    # Get this client's training data
    client_train_indices = client_indices[client_id % num_clients]
    train_subset = Subset(train_dataset, client_train_indices)

    # For test data, use a simple uniform split
    test_indices = np.arange(len(test_dataset))
    np.random.shuffle(test_indices)
    test_split_size = len(test_dataset) // num_clients
    start_idx = client_id * test_split_size
    end_idx = start_idx + test_split_size if client_id < num_clients - 1 else len(test_dataset)
    client_test_indices = test_indices[start_idx:end_idx]
    test_subset = Subset(test_dataset, client_test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues in containers
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Client {client_id}: MNIST data loaded - {len(train_subset)} train, {len(test_subset)} test samples")

    return train_loader, test_loader


def get_cifar10_client_data(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    alpha: float = 0.5,
    data_path: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 data partition for a specific client.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        batch_size: Batch size for data loaders
        alpha: Non-IID parameter for Dirichlet distribution
        data_path: Path to store dataset

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Data transformations with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Ensure data directory exists
    Path(data_path).mkdir(parents=True, exist_ok=True)

    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )

    # Set deterministic seed
    np.random.seed(42)

    # Partition training data
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    client_indices = dirichlet_partition(train_labels, num_clients, alpha)

    # Get this client's data
    client_train_indices = client_indices[client_id % num_clients]
    train_subset = Subset(train_dataset, client_train_indices)

    # Test data split
    test_indices = np.arange(len(test_dataset))
    np.random.shuffle(test_indices)
    test_split_size = len(test_dataset) // num_clients
    start_idx = client_id * test_split_size
    end_idx = start_idx + test_split_size if client_id < num_clients - 1 else len(test_dataset)
    client_test_indices = test_indices[start_idx:end_idx]
    test_subset = Subset(test_dataset, client_test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Client {client_id}: CIFAR-10 data loaded - {len(train_subset)} train, {len(test_subset)} test samples")

    return train_loader, test_loader


def get_synthetic_client_data(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    dataset_size: int = 1000,
    input_dim: int = 784,
    num_classes: int = 10
) -> Tuple[DataLoader, DataLoader]:
    """
    Get synthetic data partition for a specific client.

    Args:
        client_id: Client identifier
        num_clients: Total number of clients
        batch_size: Batch size for data loaders
        dataset_size: Size of synthetic dataset
        input_dim: Input dimension
        num_classes: Number of classes

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create synthetic dataset
    full_dataset = SyntheticDataset(dataset_size, input_dim, num_classes)

    # Split data among clients
    samples_per_client = dataset_size // num_clients
    start_idx = client_id * samples_per_client
    end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else dataset_size

    client_indices = list(range(start_idx, end_idx))
    client_subset = Subset(full_dataset, client_indices)

    # Split into train/test (80/20)
    subset_size = len(client_subset)
    train_size = int(0.8 * subset_size)
    test_size = subset_size - train_size

    train_indices = client_indices[:train_size]
    test_indices = client_indices[train_size:]

    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Client {client_id}: Synthetic data loaded - {len(train_subset)} train, {len(test_subset)} test samples")

    return train_loader, test_loader


def get_client_data(
    dataset: str,
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    alpha: float = 0.5,
    data_path: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get client data for any supported dataset.

    Args:
        dataset: Dataset name ('mnist', 'cifar10', 'synthetic')
        client_id: Client identifier
        num_clients: Total number of clients
        batch_size: Batch size
        alpha: Non-IID parameter
        data_path: Data storage path

    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset == "mnist":
        return get_mnist_client_data(client_id, num_clients, batch_size, alpha, data_path)
    elif dataset == "cifar10":
        return get_cifar10_client_data(client_id, num_clients, batch_size, alpha, data_path)
    elif dataset == "synthetic":
        return get_synthetic_client_data(client_id, num_clients, batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")