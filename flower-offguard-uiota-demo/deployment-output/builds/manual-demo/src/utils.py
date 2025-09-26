"""
Utility functions for federated learning demo
"""

import base64
import logging
import pickle
from typing import Any, Dict, List
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
from . import guard

logger = logging.getLogger(__name__)


def serialize_parameters(parameters: List[np.ndarray]) -> bytes:
    """
    Serialize model parameters to bytes for signing/verification.

    Args:
        parameters: List of numpy arrays (model weights)

    Returns:
        Serialized parameters as bytes
    """
    try:
        return pickle.dumps(parameters)
    except Exception as e:
        logger.error(f"Failed to serialize parameters: {e}")
        raise


def deserialize_parameters(data: bytes) -> List[np.ndarray]:
    """
    Deserialize model parameters from bytes.

    Args:
        data: Serialized parameters

    Returns:
        List of numpy arrays
    """
    try:
        return pickle.loads(data)
    except Exception as e:
        logger.error(f"Failed to deserialize parameters: {e}")
        raise


def bytes_to_base64(data: bytes) -> str:
    """Convert bytes to base64 string for JSON serialization."""
    return base64.b64encode(data).decode('utf-8')


def base64_to_bytes(data: str) -> bytes:
    """Convert base64 string back to bytes."""
    return base64.b64decode(data.encode('utf-8'))


def serialize_public_key(public_key: ed25519.Ed25519PublicKey) -> str:
    """Serialize public key to base64 string."""
    key_bytes = guard.serialize_public_key(public_key)
    return bytes_to_base64(key_bytes)


def deserialize_public_key(key_str: str) -> ed25519.Ed25519PublicKey:
    """Deserialize public key from base64 string."""
    key_bytes = base64_to_bytes(key_str)
    return guard.deserialize_public_key(key_bytes)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_parameters(parameters: List[np.ndarray]) -> bool:
    """
    Validate model parameters for basic sanity checks.

    Args:
        parameters: Model parameters to validate

    Returns:
        True if parameters pass basic validation
    """
    try:
        if not parameters:
            logger.error("Empty parameters list")
            return False

        for i, param in enumerate(parameters):
            if not isinstance(param, np.ndarray):
                logger.error(f"Parameter {i} is not numpy array: {type(param)}")
                return False

            if param.size == 0:
                logger.error(f"Parameter {i} is empty")
                return False

            if not np.isfinite(param).all():
                logger.error(f"Parameter {i} contains non-finite values")
                return False

        return True

    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        return False


def compute_model_size(parameters: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compute statistics about model parameters.

    Args:
        parameters: Model parameters

    Returns:
        Dictionary with model statistics
    """
    try:
        total_params = sum(param.size for param in parameters)
        total_bytes = sum(param.nbytes for param in parameters)

        param_shapes = [param.shape for param in parameters]
        param_dtypes = [str(param.dtype) for param in parameters]

        return {
            "num_layers": len(parameters),
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "parameter_shapes": param_shapes,
            "parameter_dtypes": param_dtypes,
        }

    except Exception as e:
        logger.error(f"Failed to compute model size: {e}")
        return {"error": str(e)}


def print_model_info(parameters: List[np.ndarray], title: str = "Model Info"):
    """Print formatted model information."""
    info = compute_model_size(parameters)

    print(f"\n=== {title} ===")
    print(f"Layers: {info.get('num_layers', 'N/A')}")
    print(f"Total Parameters: {info.get('total_parameters', 'N/A'):,}")
    print(f"Memory Size: {info.get('total_mb', 'N/A'):.2f} MB")

    if 'parameter_shapes' in info:
        print("Layer Shapes:")
        for i, shape in enumerate(info['parameter_shapes']):
            print(f"  Layer {i}: {shape}")
    print()