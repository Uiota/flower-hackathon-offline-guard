"""
Dataset handling and non-IID partitioning for federated learning
"""

import logging
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float = 0.5) -> list:
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
    alpha: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST data partition for a specific client.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        batch_size: Batch size for data loaders
        alpha: Non-IID parameter for Dirichlet distribution

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
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
        num_workers=0  # Avoid multiprocessing issues
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
    alpha: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 data partition for a specific client.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        batch_size: Batch size for data loaders
        alpha: Non-IID parameter for Dirichlet distribution

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

    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
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