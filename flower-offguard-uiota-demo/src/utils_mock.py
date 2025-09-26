"""
Mock utils module for testing without numpy
"""

import random
from typing import List, Any

def normalize_data(data: List[float]) -> List[float]:
    """Mock data normalization without numpy."""
    if not data:
        return data

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5 if variance > 0 else 1.0

    return [(x - mean) / std for x in data]

def generate_random_weights(size: int) -> List[float]:
    """Generate random weights without numpy."""
    return [random.random() for _ in range(size)]

def calculate_accuracy(predictions: List[float], targets: List[float]) -> float:
    """Calculate mock accuracy without numpy."""
    if len(predictions) != len(targets):
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < 0.1)
    return correct / len(predictions) if predictions else 0.0

def aggregate_weights(weights_list: List[List[float]]) -> List[float]:
    """Aggregate weights without numpy."""
    if not weights_list:
        return []

    num_weights = len(weights_list[0])
    aggregated = []

    for i in range(num_weights):
        avg_weight = sum(weights[i] for weights in weights_list) / len(weights_list)
        aggregated.append(avg_weight)

    return aggregated

def validate_weights(weights: List[float]) -> bool:
    """Validate weights are finite and reasonable."""
    return all(isinstance(w, (int, float)) and -1000 < w < 1000 for w in weights)

class MockDataset:
    """Mock dataset class."""

    def __init__(self, size: int = 100):
        self.size = size
        self.data = [random.random() for _ in range(size)]
        self.targets = [random.randint(0, 1) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def create_mock_dataset(train: bool = True) -> MockDataset:
    """Create a mock dataset."""
    size = 800 if train else 200
    return MockDataset(size)