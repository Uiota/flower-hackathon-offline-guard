"""
Utility functions for the federated learning application
"""

import json
import pickle
import base64
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class Config:
    """Configuration management."""

    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def update(self, config_dict: Dict) -> None:
        """Update configuration with dictionary."""
        self._config.update(config_dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self._config.copy()

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=2)


class MetricsCollector:
    """Collect and manage training metrics."""

    def __init__(self):
        self.metrics = {}

    def add_metric(self, name: str, value: float, round_num: int, client_id: Optional[str] = None):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'round': round_num,
            'client_id': client_id,
            'timestamp': time.time()
        })

    def get_metrics(self, name: str) -> List[Dict]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def get_latest_metric(self, name: str) -> Optional[Dict]:
        """Get latest value for a metric."""
        values = self.get_metrics(name)
        return values[-1] if values else None

    def get_round_metrics(self, round_num: int) -> Dict[str, List[Dict]]:
        """Get all metrics for a specific round."""
        round_metrics = {}
        for name, values in self.metrics.items():
            round_values = [v for v in values if v['round'] == round_num]
            if round_values:
                round_metrics[name] = round_values
        return round_metrics

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return self.metrics.copy()


def serialize_parameters(parameters: List[np.ndarray]) -> bytes:
    """Serialize model parameters to bytes."""
    return pickle.dumps(parameters)


def deserialize_parameters(data: bytes) -> List[np.ndarray]:
    """Deserialize model parameters from bytes."""
    return pickle.loads(data)


def bytes_to_base64(data: bytes) -> str:
    """Convert bytes to base64 string."""
    return base64.b64encode(data).decode('utf-8')


def base64_to_bytes(data: str) -> bytes:
    """Convert base64 string to bytes."""
    return base64.b64decode(data.encode('utf-8'))


def hash_parameters(parameters: List[np.ndarray]) -> str:
    """Compute hash of model parameters."""
    serialized = serialize_parameters(parameters)
    return hashlib.sha256(serialized).hexdigest()


def generate_keypair() -> Tuple[bytes, bytes]:
    """Generate RSA key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return private_pem, public_pem


def sign_data(private_key_pem: bytes, data: bytes) -> bytes:
    """Sign data with private key."""
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None,
        backend=default_backend()
    )

    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return signature


def verify_signature(public_key_pem: bytes, signature: bytes, data: bytes) -> bool:
    """Verify signature with public key."""
    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )

        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return False


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> Any:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_model(model: torch.nn.Module, file_path: str) -> None:
    """Save PyTorch model."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), file_path)


def load_model(model: torch.nn.Module, file_path: str) -> torch.nn.Module:
    """Load PyTorch model."""
    model.load_state_dict(torch.load(file_path, map_location='cpu'))
    return model


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # Setup file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def get_device() -> torch.device:
    """Get the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.3f}s"