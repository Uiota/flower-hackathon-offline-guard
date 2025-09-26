"""
Custom Federated Learning Strategies with Off-Guard Security Integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import numpy as np

from . import guard
from . import mesh_sync
from . import utils

logger = logging.getLogger(__name__)


class OffGuardFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with Off-Guard security and mesh integration."""

    def __init__(self, server_keypair: Tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_keypair = server_keypair
        self.round_number = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates with signature verification."""
        logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")

        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")

        # Verify signatures and collect valid updates
        verified_results = []
        signature_failures = 0

        for client_proxy, fit_res in results:
            try:
                # Extract signature from metrics
                if "signature" in fit_res.metrics and "client_public_key" in fit_res.metrics:
                    signature = utils.base64_to_bytes(fit_res.metrics["signature"])
                    client_pubkey = utils.deserialize_public_key(fit_res.metrics["client_public_key"])

                    # Verify signature
                    params_bytes = utils.serialize_parameters(parameters_to_ndarrays(fit_res.parameters))
                    if guard.verify_blob(client_pubkey, params_bytes, signature):
                        verified_results.append((client_proxy, fit_res))
                        logger.debug(f"Signature verified for client {client_proxy.cid}")
                    else:
                        logger.error(f"Signature verification failed for client {client_proxy.cid}")
                        signature_failures += 1
                else:
                    # No signature - accept if security is disabled
                    logger.warning(f"No signature from client {client_proxy.cid}")
                    verified_results.append((client_proxy, fit_res))

            except Exception as e:
                logger.error(f"Security check failed for client {client_proxy.cid}: {e}")
                signature_failures += 1

        if signature_failures > 0:
            logger.warning(f"Round {server_round}: {signature_failures} signature verification failures")

        if not verified_results:
            logger.error(f"Round {server_round}: No valid updates received!")
            return None, {}

        # Proceed with standard FedAvg aggregation on verified results
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, verified_results, failures
        )

        # Add security metrics
        metrics["verified_clients"] = len(verified_results)
        metrics["signature_failures"] = signature_failures
        metrics["security_ratio"] = len(verified_results) / (len(results) + signature_failures) if results else 0

        # Sign aggregated model for distribution
        if aggregated_parameters and self.server_keypair:
            try:
                params_bytes = utils.serialize_parameters(parameters_to_ndarrays(aggregated_parameters))
                server_signature = guard.sign_blob(self.server_keypair[0], params_bytes)
                metrics["server_signature"] = utils.bytes_to_base64(server_signature)
            except Exception as e:
                logger.error(f"Failed to sign aggregated model: {e}")

        logger.info(f"Round {server_round}: Aggregation complete - {len(verified_results)} verified updates")

        return aggregated_parameters, metrics


class OffGuardFedProx(fl.server.strategy.FedProx):
    """FedProx with Off-Guard security integration."""

    def __init__(self, server_keypair: Tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_keypair = server_keypair

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with FedProx algorithm and signature verification."""
        logger.info(f"Round {server_round}: FedProx aggregating {len(results)} client updates")

        # Verify signatures (same logic as OffGuardFedAvg)
        verified_results = []
        signature_failures = 0

        for client_proxy, fit_res in results:
            try:
                if "signature" in fit_res.metrics and "client_public_key" in fit_res.metrics:
                    signature = utils.base64_to_bytes(fit_res.metrics["signature"])
                    client_pubkey = utils.deserialize_public_key(fit_res.metrics["client_public_key"])

                    params_bytes = utils.serialize_parameters(parameters_to_ndarrays(fit_res.parameters))
                    if guard.verify_blob(client_pubkey, params_bytes, signature):
                        verified_results.append((client_proxy, fit_res))
                    else:
                        logger.error(f"FedProx: Signature verification failed for client {client_proxy.cid}")
                        signature_failures += 1
                else:
                    verified_results.append((client_proxy, fit_res))

            except Exception as e:
                logger.error(f"FedProx security check failed for client {client_proxy.cid}: {e}")
                signature_failures += 1

        if not verified_results:
            return None, {}

        # Use parent FedProx aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, verified_results, failures
        )

        # Add security metrics
        metrics["verified_clients"] = len(verified_results)
        metrics["signature_failures"] = signature_failures

        # Sign aggregated model
        if aggregated_parameters and self.server_keypair:
            try:
                params_bytes = utils.serialize_parameters(parameters_to_ndarrays(aggregated_parameters))
                server_signature = guard.sign_blob(self.server_keypair[0], params_bytes)
                metrics["server_signature"] = utils.bytes_to_base64(server_signature)
            except Exception as e:
                logger.error(f"Failed to sign FedProx aggregated model: {e}")

        return aggregated_parameters, metrics