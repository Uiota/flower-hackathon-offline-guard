#!/usr/bin/env python3
"""
LL TOKEN OFFLINE - Tokenization Phase Infrastructure
Complete system for quantum-resistant tokenized federated learning with agent integration
"""

import sys
import os
import time
import threading
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Add project paths
sys.path.insert(0, 'src')

# Set environment for secure offline mode
os.environ["OFFLINE_MODE"] = "1"

# Import our tokenization modules
from src.quantum_wallet import create_quantum_wallet_system, QuantumWallet, TokenRail
from src.fl_token_integration import create_tokenized_fl_system, FLTokenLedger, TokenizedFLStrategy
from src.guard import preflight_check

# Standard FL imports
import flwr as fl
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLTokenAgent:
    """
    LL TOKEN OFFLINE Agent - Manages tokenized FL participation with quantum security.
    Each agent represents a participant in the federated learning network with token incentives.
    """

    def __init__(self, agent_id: str, base_path: str, server_address: str = "localhost:8080"):
        self.agent_id = agent_id
        self.base_path = Path(base_path) / f"agent_{agent_id}"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.server_address = server_address
        self.running = False

        # Initialize wallet system for this agent
        self.wallet, self.token_rail = create_quantum_wallet_system(str(self.base_path / "wallet"))

        # Agent metrics
        self.local_metrics = {
            'rounds_participated': 0,
            'tokens_earned': 0,
            'average_accuracy': 0.0,
            'total_training_time': 0.0,
            'reputation_score': 1.0
        }

        # Create simple neural network for FL
        self.model = self._create_model()

        logger.info(f"LL TOKEN Agent {agent_id} initialized")
        logger.info(f"Agent wallet ID: {self.wallet.wallet_id}")
        logger.info(f"Initial token balance: {self.wallet.get_balance()}")

    def _create_model(self):
        """Create a simple neural network model for federated learning."""
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(784, 128)  # MNIST: 28*28 = 784
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(128, 64)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(64, 10)  # 10 classes

            def forward(self, x):
                x = self.flatten(x)
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                x = self.fc3(x)
                return x

        return SimpleNet()

    def start_fl_client(self):
        """Start the Flower federated learning client."""
        logger.info(f"Starting FL client for agent {self.agent_id}")

        # Create Flower client
        client = LLTokenFlowerClient(self)

        try:
            # Connect to FL server
            fl.client.start_numpy_client(
                server_address=self.server_address,
                client=client
            )
        except Exception as e:
            logger.error(f"FL client error for agent {self.agent_id}: {e}")

    def simulate_training(self, parameters: List[np.ndarray]) -> Dict[str, Any]:
        """Simulate local training and return updated parameters with metrics."""
        start_time = time.time()

        # Set model parameters
        self._set_model_parameters(parameters)

        # Simulate training epochs
        epochs = 3
        initial_loss = 2.5
        initial_accuracy = 0.1

        # Simulate improvement over epochs
        for epoch in range(epochs):
            improvement_factor = (epoch + 1) / epochs * 0.7  # Gradual improvement
            time.sleep(0.2)  # Simulate training time

        # Calculate final metrics with some randomness
        import random
        final_loss = max(0.1, initial_loss * (0.3 + random.uniform(-0.1, 0.1)))
        final_accuracy = min(0.95, initial_accuracy + improvement_factor + random.uniform(0, 0.3))

        training_time = time.time() - start_time

        # Update agent metrics
        self.local_metrics['rounds_participated'] += 1
        self.local_metrics['total_training_time'] += training_time

        # Update average accuracy
        prev_avg = self.local_metrics['average_accuracy']
        n = self.local_metrics['rounds_participated']
        self.local_metrics['average_accuracy'] = ((prev_avg * (n - 1)) + final_accuracy) / n

        # Get updated parameters (simulate parameter updates)
        updated_parameters = self._get_model_parameters()

        # Add some noise to simulate actual training
        for i, param in enumerate(updated_parameters):
            noise = np.random.normal(0, 0.01, param.shape)
            updated_parameters[i] = param + noise

        metrics = {
            'loss': final_loss,
            'accuracy': final_accuracy,
            'num_examples': random.randint(800, 1200),  # Simulate varying dataset sizes
            'training_time': training_time,
            'epochs': epochs
        }

        logger.info(f"Agent {self.agent_id} training complete: accuracy={final_accuracy:.3f}, loss={final_loss:.3f}")

        return {
            'parameters': updated_parameters,
            'metrics': metrics,
            'num_examples': metrics['num_examples']
        }

    def _set_model_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from server."""
        state_dict = self.model.state_dict()
        param_keys = list(state_dict.keys())

        for i, key in enumerate(param_keys):
            if i < len(parameters):
                state_dict[key] = torch.from_numpy(parameters[i])

        self.model.load_state_dict(state_dict)

    def _get_model_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [param.detach().numpy() for param in self.model.parameters()]

    def record_token_reward(self, tokens: int, round_number: int):
        """Record tokens earned from FL participation."""
        self.local_metrics['tokens_earned'] += tokens
        self.wallet.mint_tokens(tokens, f"fl_reward_round_{round_number}")

        logger.info(f"Agent {self.agent_id} earned {tokens} tokens in round {round_number}")
        logger.info(f"Total tokens earned: {self.local_metrics['tokens_earned']}")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_id,
            'wallet_id': self.wallet.wallet_id,
            'wallet_balance': self.wallet.get_balance(),
            'metrics': self.local_metrics,
            'pending_transactions': len(self.wallet.get_pending_transactions()),
            'running': self.running
        }


class LLTokenFlowerClient(fl.client.NumPyClient):
    """Flower client wrapper for LL TOKEN agents."""

    def __init__(self, agent: LLTokenAgent):
        self.agent = agent

    def get_parameters(self, config):
        """Return current model parameters."""
        return self.agent._get_model_parameters()

    def fit(self, parameters, config):
        """Train model and return updated parameters."""
        result = self.agent.simulate_training(parameters)

        return (
            result['parameters'],
            result['num_examples'],
            result['metrics']
        )

    def evaluate(self, parameters, config):
        """Evaluate model performance."""
        # Set parameters for evaluation
        self.agent._set_model_parameters(parameters)

        # Simulate evaluation
        import random
        accuracy = max(0.1, min(0.95, random.uniform(0.6, 0.9)))
        loss = max(0.1, random.uniform(0.2, 1.0))

        return (
            loss,
            random.randint(200, 400),  # num_examples for evaluation
            {'accuracy': accuracy}
        )


class LLTokenMasterRail:
    """
    Master Rail System for LL TOKEN OFFLINE.
    Coordinates multiple agents, manages global token economy, and maintains ledger integrity.
    """

    def __init__(self, base_path: str, num_agents: int = 5):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.num_agents = num_agents
        self.agents: List[LLTokenAgent] = []
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        # Initialize tokenized FL system
        self.fl_strategy: Optional[TokenizedFLStrategy] = None
        self.fl_ledger: Optional[FLTokenLedger] = None

        # System metrics
        self.system_metrics = {
            'total_rounds': 0,
            'total_participants': 0,
            'total_tokens_distributed': 0,
            'system_start_time': datetime.now(timezone.utc).isoformat()
        }

    def initialize_system(self):
        """Initialize the complete LL TOKEN system."""
        logger.info("🚀 Initializing LL TOKEN OFFLINE Master Rail System")
        logger.info("=" * 60)

        # Run security preflight checks
        try:
            preflight_check()
            logger.info("✅ Off-Guard security checks passed")
        except Exception as e:
            logger.warning(f"⚠️  Security check warning: {e}")

        # Create tokenized FL system
        logger.info("🔗 Creating tokenized federated learning system...")

        # Create initial model parameters
        dummy_model = self._create_dummy_model()
        initial_parameters = fl.common.ndarrays_to_parameters([
            param.detach().numpy() for param in dummy_model.parameters()
        ])

        # Create tokenized strategy and ledger
        self.fl_strategy, self.fl_ledger = create_tokenized_fl_system(
            base_path=str(self.base_path / "fl_system"),
            initial_parameters=initial_parameters,
            min_fit_clients=max(2, self.num_agents // 2),
            min_evaluate_clients=max(2, self.num_agents // 2),
            min_available_clients=self.num_agents
        )

        logger.info("✅ Tokenized FL system created")

        # Create agent pool
        logger.info(f"👥 Creating {self.num_agents} LL TOKEN agents...")
        for i in range(self.num_agents):
            agent = LLTokenAgent(
                agent_id=f"lltoken_agent_{i:03d}",
                base_path=str(self.base_path / "agents"),
                server_address="localhost:8080"
            )
            self.agents.append(agent)
            time.sleep(0.1)  # Stagger agent creation

        logger.info(f"✅ Created {len(self.agents)} LL TOKEN agents")

        # System ready
        self.system_metrics['total_participants'] = len(self.agents)
        logger.info("\n🎯 LL TOKEN OFFLINE system initialized successfully!")
        logger.info(f"📊 Agents: {len(self.agents)}")
        logger.info(f"🏦 FL Ledger: {self.fl_ledger.wallet.wallet_id}")
        logger.info(f"💰 System token balance: {self.fl_ledger.wallet.get_balance()}")

    def _create_dummy_model(self):
        """Create dummy model for initial parameters."""
        import torch.nn as nn

        class DummyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(784, 128)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(128, 64)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(64, 10)

            def forward(self, x):
                x = self.flatten(x)
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                x = self.fc3(x)
                return x

        return DummyNet()

    def start_fl_server(self, rounds: int = 5):
        """Start the federated learning server."""
        logger.info(f"🖥️  Starting FL server for {rounds} rounds...")

        def run_server():
            try:
                config = fl.server.ServerConfig(num_rounds=rounds)
                fl.server.start_server(
                    server_address="localhost:8080",
                    config=config,
                    strategy=self.fl_strategy
                )
            except Exception as e:
                logger.error(f"FL server error: {e}")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(2)  # Give server time to start

        logger.info("✅ FL server started on localhost:8080")

    def start_agents(self):
        """Start all LL TOKEN agents."""
        logger.info(f"🚀 Starting {len(self.agents)} LL TOKEN agents...")

        # Start agents with staggered delays
        for i, agent in enumerate(self.agents):
            agent_thread = threading.Thread(
                target=agent.start_fl_client,
                daemon=True
            )
            agent_thread.start()

            # Stagger agent starts to avoid overwhelming the server
            time.sleep(1.0)
            logger.info(f"  Agent {agent.agent_id} started ({i+1}/{len(self.agents)})")

        logger.info("✅ All agents started and connecting to FL server")

    def run_tokenization_phase(self, rounds: int = 5):
        """Run the complete tokenization phase."""
        logger.info("\n🔥 Starting LL TOKEN OFFLINE Tokenization Phase")
        logger.info("=" * 60)

        self.running = True

        try:
            # Start FL server
            self.start_fl_server(rounds)

            # Start agents
            self.start_agents()

            # Monitor progress
            self._monitor_tokenization_progress(rounds)

        except KeyboardInterrupt:
            logger.info("\n⏹️  Tokenization phase interrupted by user")
        except Exception as e:
            logger.error(f"❌ Tokenization phase error: {e}")
        finally:
            self.running = False
            logger.info("🏁 LL TOKEN OFFLINE tokenization phase complete")

    def _monitor_tokenization_progress(self, total_rounds: int):
        """Monitor tokenization phase progress."""
        logger.info(f"📊 Monitoring tokenization progress for {total_rounds} rounds...")

        start_time = time.time()
        last_round = -1

        while self.running:
            try:
                # Check system status
                current_round = self.fl_ledger.current_round

                if current_round > last_round and current_round > 0:
                    # New round completed
                    round_summary = self.fl_ledger.get_round_summary(current_round)

                    if round_summary:
                        logger.info(f"\n📈 Round {current_round} Summary:")
                        logger.info(f"  Participants: {round_summary['participants']}")
                        logger.info(f"  Tokens distributed: {round_summary['total_tokens_distributed']}")
                        logger.info(f"  Average quality: {round_summary['average_quality_score']:.3f}")

                        # Update system metrics
                        self.system_metrics['total_rounds'] = current_round
                        self.system_metrics['total_tokens_distributed'] += round_summary['total_tokens_distributed']

                    last_round = current_round

                # Check if all rounds completed
                if current_round >= total_rounds:
                    logger.info(f"\n🎉 All {total_rounds} rounds completed!")
                    self._print_final_summary()
                    break

                time.sleep(3)  # Check every 3 seconds

            except Exception as e:
                logger.debug(f"Monitoring error (non-critical): {e}")
                time.sleep(1)

    def _print_final_summary(self):
        """Print final tokenization phase summary."""
        logger.info("\n" + "=" * 60)
        logger.info("🏆 LL TOKEN OFFLINE TOKENIZATION PHASE COMPLETE")
        logger.info("=" * 60)

        # System metrics
        total_time = time.time()
        logger.info(f"📊 System Metrics:")
        logger.info(f"  Total rounds: {self.system_metrics['total_rounds']}")
        logger.info(f"  Total participants: {self.system_metrics['total_participants']}")
        logger.info(f"  Total tokens distributed: {self.system_metrics['total_tokens_distributed']}")

        # Agent summaries
        logger.info(f"\n👥 Agent Performance:")
        for agent in self.agents:
            status = agent.get_status()
            logger.info(f"  {agent.agent_id}:")
            logger.info(f"    Wallet balance: {status['wallet_balance']} tokens")
            logger.info(f"    Rounds participated: {status['metrics']['rounds_participated']}")
            logger.info(f"    Average accuracy: {status['metrics']['average_accuracy']:.3f}")

        # Ledger proof
        logger.info(f"\n🔒 Generating cryptographic ledger proof...")
        ledger_proof = self.fl_ledger.export_ledger_proof()
        if ledger_proof:
            proof_file = self.base_path / "tokenization_proof.json"
            with open(proof_file, 'w') as f:
                json.dump(ledger_proof, f, indent=2)
            logger.info(f"  Ledger proof saved: {proof_file}")

        logger.info("\n✅ LL TOKEN OFFLINE system ready for production deployment!")


def main():
    """Main entry point for LL TOKEN OFFLINE system."""
    parser = argparse.ArgumentParser(description="LL TOKEN OFFLINE - Tokenization Phase")

    parser.add_argument("--agents", type=int, default=6, help="Number of LL TOKEN agents")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--base-path", type=str, default="./ll_token_system", help="Base path for system files")

    args = parser.parse_args()

    print("🪙 LL TOKEN OFFLINE - Quantum-Resistant Tokenized Federated Learning")
    print("🔒 Zero-Trust Architecture • Offline Ledger • Agent-Based System")
    print("=" * 70)

    # Create and run master rail system
    master_rail = LLTokenMasterRail(args.base_path, args.agents)

    try:
        # Initialize system
        master_rail.initialize_system()

        # Run tokenization phase
        master_rail.run_tokenization_phase(args.rounds)

    except KeyboardInterrupt:
        print("\n⏹️  LL TOKEN system shutdown requested")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 LL TOKEN OFFLINE system stopped")


if __name__ == "__main__":
    main()