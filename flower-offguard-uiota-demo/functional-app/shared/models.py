"""
Neural Network Models for Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import json


class SmallCNN(nn.Module):
    """Small CNN for MNIST dataset."""

    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 dataset."""

    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class LinearClassifier(nn.Module):
    """Simple linear classifier for testing."""

    def __init__(self, input_size=784, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model(model_name: str, dataset: str) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_name: Name of the model ('cnn', 'linear')
        dataset: Dataset name ('mnist', 'cifar10')

    Returns:
        PyTorch model instance
    """
    if dataset == "mnist":
        if model_name == "cnn":
            return SmallCNN(num_classes=10)
        elif model_name == "linear":
            return LinearClassifier(input_size=784, num_classes=10)
    elif dataset == "cifar10":
        if model_name == "cnn":
            return CIFAR10CNN(num_classes=10)
        elif model_name == "linear":
            return LinearClassifier(input_size=3072, num_classes=10)

    raise ValueError(f"Unsupported model_name={model_name} for dataset={dataset}")


def model_to_dict(model: nn.Module) -> Dict[str, Any]:
    """Convert model state dict to serializable format."""
    return {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}


def dict_to_model(model: nn.Module, state_dict: Dict[str, Any]) -> nn.Module:
    """Load model state from serializable format."""
    tensor_dict = {k: torch.tensor(v) for k, v in state_dict.items()}
    model.load_state_dict(tensor_dict)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)