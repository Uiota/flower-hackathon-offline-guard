"""
Flower Labs Integration for LL TOKEN OFFLINE Ledger
Combines federated learning with tokenized incentive mechanisms
"""

import json
import logging
import time
import threading
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import flwr as fl
import numpy as np
from flwr.common import NDArrays, Parameters, Scalar
from flwr.server.strategy import Strategy

from .quantum_wallet import QuantumWallet, TokenRail, create_quantum_wallet_system
from .guard import preflight_check, new_keypair, sign_blob, verify_blob

logger = logging.getLogger(__name__)


@dataclass
class FLTokenConfig:
    """Configuration for FL token integration."""
    base_reward_per_round: int = 100  # Base tokens per round
    quality_multiplier_max: float = 2.0  # Max multiplier for high quality updates
    participation_bonus: int = 50  # Bonus for consistent participation
    validator_reward: int = 25  # Reward for validating other participants
    min_quality_threshold: float = 0.1  # Minimum quality to receive rewards


@dataclass
class ParticipantMetrics:
    """Track participant performance and token rewards."""
    client_id: str
    rounds_participated: int = 0
    total_tokens_earned: int = 0
    average_contribution_quality: float = 0.0
    last_participation: Optional[datetime] = None
    reputation_score: float = 1.0  # 0-2 scale
    consecutive_participations: int = 0


class FLTokenLedger:
    """
    Offline ledger for tracking FL participant contributions and token rewards.
    Integrates with Flower Labs framework for incentivized federated learning.
    """

    def __init__(self, ledger_path: str, wallet_system: Tuple[QuantumWallet, TokenRail] = None):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.mkdir(parents=True, exist_ok=True)

        self.config = FLTokenConfig()

        # Initialize or use provided wallet system
        if wallet_system:
            self.wallet, self.token_rail = wallet_system
        else:
            self.wallet, self.token_rail = create_quantum_wallet_system(str(self.ledger_path))

        # Participant tracking
        self.participants: Dict[str, ParticipantMetrics] = {}
        self.round_contributions: Dict[int, List[Dict]] = {}
        self.current_round: int = 0

        # Security
        self.ledger_keypair = new_keypair()

        # Load existing state
        self._load_ledger_state()

        logger.info("FL Token Ledger initialized")

    def _load_ledger_state(self):
        """Load ledger state from storage."""
        state_file = self.ledger_path / "fl_token_ledger.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)

                # Restore participants
                for client_id, data in state_data.get('participants', {}).items():
                    self.participants[client_id] = ParticipantMetrics(
                        client_id=client_id,
                        rounds_participated=data['rounds_participated'],
                        total_tokens_earned=data['total_tokens_earned'],
                        average_contribution_quality=data['average_contribution_quality'],
                        last_participation=datetime.fromisoformat(data['last_participation']) if data.get('last_participation') else None,
                        reputation_score=data.get('reputation_score', 1.0),
                        consecutive_participations=data.get('consecutive_participations', 0)
                    )

                self.round_contributions = state_data.get('round_contributions', {})
                # Convert string keys back to integers
                self.round_contributions = {int(k): v for k, v in self.round_contributions.items()}

                self.current_round = state_data.get('current_round', 0)
                logger.info(f"Loaded FL token ledger state: {len(self.participants)} participants")

            except Exception as e:
                logger.warning(f"Failed to load ledger state: {e}")

    def _save_ledger_state(self):
        """Save ledger state to storage."""
        state_file = self.ledger_path / "fl_token_ledger.json"

        try:
            state_data = {
                'participants': {},
                'round_contributions': {},
                'current_round': self.current_round,
                'last_update': datetime.now(timezone.utc).isoformat(),
                'config': asdict(self.config)
            }

            # Serialize participants
            for client_id, metrics in self.participants.items():
                participant_data = asdict(metrics)
                if participant_data['last_participation']:
                    participant_data['last_participation'] = metrics.last_participation.isoformat()
                state_data['participants'][client_id] = participant_data

            # Serialize round contributions (convert int keys to strings for JSON)
            state_data['round_contributions'] = {str(k): v for k, v in self.round_contributions.items()}

            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save ledger state: {e}")

    def start_round(self, round_number: int):
        """Initialize a new FL training round."""
        self.current_round = round_number
        self.round_contributions[round_number] = []

        logger.info(f"Started FL round {round_number} for token rewards")

    def record_contribution(
        self,
        client_id: str,
        round_number: int,
        model_update: NDArrays,
        training_metrics: Dict[str, Scalar],
        validation_results: Dict[str, Scalar] = None
    ) -> Dict[str, Any]:
        """
        Record a client's contribution to the FL round and calculate token rewards.

        Returns contribution record with calculated rewards.
        """

        # Initialize participant if new
        if client_id not in self.participants:
            self.participants[client_id] = ParticipantMetrics(client_id=client_id)

        participant = self.participants[client_id]

        # Calculate contribution quality based on training metrics
        quality_score = self._calculate_quality_score(training_metrics, validation_results)

        # Create contribution record
        contribution = {
            'client_id': client_id,
            'round_number': round_number,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'quality_score': quality_score,
            'training_metrics': dict(training_metrics) if training_metrics else {},
            'validation_results': dict(validation_results) if validation_results else {},
            'model_update_hash': self._hash_model_update(model_update),
            'tokens_earned': 0  # Will be calculated after aggregation
        }

        # Add to round contributions
        if round_number not in self.round_contributions:
            self.round_contributions[round_number] = []

        self.round_contributions[round_number].append(contribution)

        # Update participant metrics
        participant.rounds_participated += 1
        participant.last_participation = datetime.now(timezone.utc)

        # Calculate consecutive participations
        if round_number > 0 and any(c['client_id'] == client_id for c in self.round_contributions.get(round_number - 1, [])):
            participant.consecutive_participations += 1
        else:
            participant.consecutive_participations = 1

        # Update average quality
        prev_avg = participant.average_contribution_quality
        n = participant.rounds_participated
        participant.average_contribution_quality = ((prev_avg * (n - 1)) + quality_score) / n

        logger.info(f"Recorded contribution from {client_id} in round {round_number}, quality: {quality_score:.3f}")

        # Save state
        self._save_ledger_state()

        return contribution

    def finalize_round_rewards(self, round_number: int, aggregation_metrics: Dict[str, Any] = None) -> Dict[str, int]:
        """
        Finalize token rewards for all participants in a round after aggregation.

        Returns dict of client_id -> tokens_earned
        """
        if round_number not in self.round_contributions:
            logger.warning(f"No contributions found for round {round_number}")
            return {}

        contributions = self.round_contributions[round_number]
        rewards = {}

        # Calculate rewards for each contribution
        for contribution in contributions:
            client_id = contribution['client_id']
            quality_score = contribution['quality_score']

            # Base reward
            base_reward = self.config.base_reward_per_round

            # Quality multiplier
            quality_multiplier = min(
                self.config.quality_multiplier_max,
                max(0.1, quality_score)
            )

            # Participation bonus
            participant = self.participants[client_id]
            participation_bonus = 0
            if participant.consecutive_participations >= 3:
                participation_bonus = self.config.participation_bonus

            # Reputation multiplier
            reputation_multiplier = participant.reputation_score

            # Calculate total tokens
            total_tokens = int((base_reward * quality_multiplier + participation_bonus) * reputation_multiplier)

            # Ensure minimum quality threshold
            if quality_score < self.config.min_quality_threshold:
                total_tokens = max(1, total_tokens // 10)  # Minimal reward for poor quality

            # Update contribution record
            contribution['tokens_earned'] = total_tokens
            rewards[client_id] = total_tokens

            # Update participant totals
            participant.total_tokens_earned += total_tokens

            logger.info(f"Calculated rewards for {client_id}: {total_tokens} tokens (quality: {quality_score:.3f})")

        # Process token transactions
        self._process_token_rewards(rewards, round_number)

        # Save state
        self._save_ledger_state()

        return rewards

    def _calculate_quality_score(
        self,
        training_metrics: Dict[str, Scalar],
        validation_results: Dict[str, Scalar] = None
    ) -> float:
        """
        Calculate contribution quality score based on training and validation metrics.

        Score ranges from 0.0 to 2.0, with 1.0 being average quality.
        """
        try:
            # Default base score
            quality_score = 1.0

            # Factor in training loss improvement
            if 'loss' in training_metrics:
                loss = float(training_metrics['loss'])
                # Lower loss is better, score inversely related
                loss_factor = max(0.1, min(2.0, 2.0 - loss))
                quality_score *= loss_factor

            # Factor in accuracy if available
            if 'accuracy' in training_metrics:
                accuracy = float(training_metrics['accuracy'])
                # Higher accuracy is better
                accuracy_factor = max(0.1, min(2.0, accuracy * 2))
                quality_score *= accuracy_factor

            # Factor in number of training examples
            if 'num_examples' in training_metrics:
                num_examples = int(training_metrics['num_examples'])
                # More examples generally means better contribution
                example_factor = min(1.5, max(0.5, num_examples / 1000))
                quality_score *= example_factor

            # Factor in validation results if available
            if validation_results:
                if 'accuracy' in validation_results:
                    val_accuracy = float(validation_results['accuracy'])
                    val_factor = max(0.8, min(1.2, val_accuracy))
                    quality_score *= val_factor

            # Normalize to reasonable range
            quality_score = max(0.0, min(2.0, quality_score))

            return quality_score

        except Exception as e:
            logger.warning(f"Failed to calculate quality score: {e}")
            return 1.0  # Default score

    def _hash_model_update(self, model_update: NDArrays) -> str:
        """Create hash of model update for integrity verification."""
        try:
            # Concatenate all arrays and hash
            update_bytes = b''
            for array in model_update:
                update_bytes += array.tobytes()

            import hashlib
            return hashlib.sha256(update_bytes).hexdigest()

        except Exception as e:
            logger.warning(f"Failed to hash model update: {e}")
            return "hash_error"

    def _process_token_rewards(self, rewards: Dict[str, int], round_number: int):
        """Process token rewards by creating transactions."""
        try:
            # Create batch of reward transactions
            transactions = []

            for client_id, tokens in rewards.items():
                if tokens > 0:
                    # Create reward transaction
                    transaction = self.wallet.create_transaction(
                        to_address=client_id,  # In practice, this would be the client's wallet address
                        amount=tokens,
                        metadata={
                            'type': 'fl_reward',
                            'round_number': round_number,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    transactions.append(transaction)

            # Submit batch to token rail
            if transactions:
                batch_id = self.token_rail.submit_transaction_batch(
                    transactions=transactions,
                    batch_metadata={
                        'type': 'fl_round_rewards',
                        'round_number': round_number,
                        'total_participants': len(rewards)
                    }
                )
                logger.info(f"Submitted reward batch {batch_id} for round {round_number}")

        except Exception as e:
            logger.error(f"Failed to process token rewards: {e}")

    def get_participant_stats(self, client_id: str = None) -> Dict[str, Any]:
        """Get statistics for a specific participant or all participants."""
        if client_id:
            if client_id in self.participants:
                return asdict(self.participants[client_id])
            else:
                return {}
        else:
            return {cid: asdict(metrics) for cid, metrics in self.participants.items()}

    def get_round_summary(self, round_number: int) -> Dict[str, Any]:
        """Get summary of a specific round."""
        if round_number not in self.round_contributions:
            return {}

        contributions = self.round_contributions[round_number]
        total_tokens = sum(c.get('tokens_earned', 0) for c in contributions)
        avg_quality = sum(c['quality_score'] for c in contributions) / len(contributions) if contributions else 0

        return {
            'round_number': round_number,
            'participants': len(contributions),
            'total_tokens_distributed': total_tokens,
            'average_quality_score': avg_quality,
            'contributions': contributions
        }

    def export_ledger_proof(self) -> Dict[str, Any]:
        """Export cryptographically signed ledger proof for audit."""
        try:
            # Create ledger summary
            ledger_proof = {
                'ledger_id': self.wallet.wallet_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_participants': len(self.participants),
                'total_rounds': len(self.round_contributions),
                'total_tokens_distributed': sum(p.total_tokens_earned for p in self.participants.values()),
                'current_round': self.current_round,
                'participants_summary': {
                    cid: {
                        'rounds_participated': p.rounds_participated,
                        'total_tokens_earned': p.total_tokens_earned,
                        'reputation_score': p.reputation_score
                    }
                    for cid, p in self.participants.items()
                }
            }

            # Sign the proof
            proof_bytes = json.dumps(ledger_proof, sort_keys=True).encode()
            signature = sign_blob(self.ledger_keypair[0], proof_bytes)

            return {
                'proof': ledger_proof,
                'signature': signature.hex(),
                'public_key': self.ledger_keypair[1].public_bytes(
                    encoding=fl.common.Encoding.Raw,
                    format=fl.common.PublicFormat.Raw
                ).hex()
            }

        except Exception as e:
            logger.error(f"Failed to create ledger proof: {e}")
            return {}


class TokenizedFLStrategy(fl.server.strategy.FedAvg):
    """
    Flower strategy that integrates token rewards for FL participants.
    Extends FedAvg with tokenization capabilities.
    """

    def __init__(self, fl_token_ledger: FLTokenLedger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fl_token_ledger = fl_token_ledger
        self.current_round = 0

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize parameters and start first round in ledger."""
        self.current_round = 0
        self.fl_token_ledger.start_round(self.current_round)
        return super().initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure fit round and update ledger."""
        self.current_round = server_round
        self.fl_token_ledger.start_round(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and record contributions in ledger."""

        # Record contributions before aggregation
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid

            # Extract training metrics
            training_metrics = fit_res.metrics or {}

            # Record contribution
            self.fl_token_ledger.record_contribution(
                client_id=client_id,
                round_number=server_round,
                model_update=fl.common.parameters_to_ndarrays(fit_res.parameters),
                training_metrics=training_metrics
            )

        # Perform standard aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Finalize rewards after aggregation
        rewards = self.fl_token_ledger.finalize_round_rewards(
            server_round,
            aggregation_metrics=aggregated_metrics
        )

        # Add reward info to aggregated metrics
        if rewards:
            aggregated_metrics = aggregated_metrics or {}
            aggregated_metrics.update({
                'total_tokens_distributed': sum(rewards.values()),
                'participants_rewarded': len(rewards)
            })

        logger.info(f"Round {server_round} complete. Distributed {sum(rewards.values())} tokens to {len(rewards)} participants")

        return aggregated_parameters, aggregated_metrics


def create_tokenized_fl_system(
    base_path: str,
    initial_parameters: Parameters,
    **strategy_kwargs
) -> Tuple[TokenizedFLStrategy, FLTokenLedger]:
    """
    Create a complete tokenized federated learning system.

    Returns the FL strategy and token ledger for integration with Flower.
    """

    # Ensure offline mode
    import os
    os.environ["OFFLINE_MODE"] = "1"

    # Run security checks
    preflight_check()

    # Create wallet system
    wallet_system = create_quantum_wallet_system(base_path)

    # Create FL token ledger
    ledger_path = Path(base_path) / "fl_token_ledger"
    fl_token_ledger = FLTokenLedger(str(ledger_path), wallet_system)

    # Create tokenized strategy
    strategy = TokenizedFLStrategy(
        fl_token_ledger=fl_token_ledger,
        initial_parameters=initial_parameters,
        **strategy_kwargs
    )

    logger.info("✅ Tokenized FL system created successfully")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Wallet ID: {wallet_system[0].wallet_id}")
    logger.info(f"Initial balance: {wallet_system[0].get_balance()} tokens")

    return strategy, fl_token_ledger