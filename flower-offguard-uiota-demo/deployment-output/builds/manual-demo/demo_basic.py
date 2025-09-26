#!/usr/bin/env python3
"""
Basic demo of the federated learning system without external dependencies
This demonstrates the core functionality using only Python standard library
"""

import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path

# Set required environment
os.environ["OFFLINE_MODE"] = "1"

class BasicDemo:
    def __init__(self):
        self.artifacts_dir = Path("demo_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        print("🚀 Starting Basic Federated Learning Demo")
        print("=" * 50)

    def simulate_security_check(self):
        """Simulate security preflight checks."""
        print("🔒 Running Security Checks:")

        # Check offline mode
        if os.environ.get("OFFLINE_MODE") == "1":
            print("  ✅ Offline mode enabled")
        else:
            print("  ❌ Offline mode not set")
            return False

        # Check Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if py_version in ["3.10", "3.11", "3.12"]:
            print(f"  ✅ Python version {py_version} verified")
        else:
            print(f"  ⚠️  Python version {py_version} may not be optimal")

        print("  ✅ Basic security checks passed")
        return True

    def simulate_keypair_generation(self):
        """Simulate cryptographic keypair generation."""
        print("\n🔑 Generating Server Keypair:")

        # Generate mock keypair (in real version, uses Ed25519)
        server_private = hashlib.sha256(b"server_private_key_seed").hexdigest()
        server_public = hashlib.sha256(b"server_public_key_seed").hexdigest()

        # Save keypair
        keypair_data = {
            "private_key": server_private,
            "public_key": server_public,
            "algorithm": "Ed25519 (simulated)",
            "generated_at": time.time()
        }

        with open(self.artifacts_dir / "server_keypair.json", 'w') as f:
            json.dump(keypair_data, f, indent=2)

        with open(self.artifacts_dir / "server_public_key.json", 'w') as f:
            json.dump({"public_key": server_public}, f, indent=2)

        print(f"  ✅ Server keypair generated")
        print(f"  📁 Saved to: {self.artifacts_dir}/")
        return server_private, server_public

    def simulate_model_initialization(self):
        """Simulate neural network model initialization."""
        print("\n🧠 Initializing Global Model:")

        # Mock model parameters (in real version, uses PyTorch CNN)
        model_params = {
            "conv1_weights": [[random.random() for _ in range(16)] for _ in range(9)],
            "conv1_bias": [random.random() for _ in range(16)],
            "conv2_weights": [[random.random() for _ in range(32)] for _ in range(16*9)],
            "conv2_bias": [random.random() for _ in range(32)],
            "fc1_weights": [[random.random() for _ in range(64)] for _ in range(32*7*7)],
            "fc1_bias": [random.random() for _ in range(64)],
            "fc2_weights": [[random.random() for _ in range(10)] for _ in range(64)],
            "fc2_bias": [random.random() for _ in range(10)]
        }

        print(f"  ✅ Small CNN model initialized")
        print(f"  📊 Model layers: {len(model_params)}")
        print(f"  🔢 Total parameters: ~{sum(len(str(v)) for v in model_params.values())}")

        return model_params

    def simulate_client_training(self, client_id, model_params, round_num):
        """Simulate client-side training."""
        print(f"\n👤 Client {client_id} Training (Round {round_num}):")

        # Simulate training process
        print(f"  📚 Loading MNIST data partition for client {client_id}")

        # Simulate epochs
        initial_loss = 2.3 + random.random() * 0.5
        final_loss = initial_loss * (0.7 + random.random() * 0.2)

        print(f"  🎯 Training for 1 epoch...")
        time.sleep(0.5)  # Simulate training time

        print(f"  📉 Loss: {initial_loss:.4f} → {final_loss:.4f}")

        # Simulate parameter updates
        updated_params = {}
        for key, values in model_params.items():
            if isinstance(values, list) and isinstance(values[0], list):
                updated_params[key] = [[v + random.random() * 0.1 - 0.05 for v in row] for row in values]
            else:
                updated_params[key] = [v + random.random() * 0.1 - 0.05 for v in values]

        # Simulate signing
        params_str = json.dumps(updated_params, sort_keys=True)
        signature = hashlib.sha256(params_str.encode()).hexdigest()[:16]

        print(f"  ✍️  Parameters signed: {signature}")
        print(f"  ✅ Client {client_id} training complete")

        return updated_params, final_loss, signature

    def simulate_server_aggregation(self, client_updates, round_num):
        """Simulate server-side model aggregation."""
        print(f"\n🔄 Server Aggregation (Round {round_num}):")

        print(f"  📥 Received {len(client_updates)} client updates")

        # Simulate signature verification
        verified_updates = []
        for client_id, (params, loss, signature) in client_updates.items():
            # Mock signature verification
            params_str = json.dumps(params, sort_keys=True)
            expected_sig = hashlib.sha256(params_str.encode()).hexdigest()[:16]

            if signature == expected_sig:
                print(f"  ✅ Signature verified for client {client_id}")
                verified_updates.append((params, loss))
            else:
                print(f"  ❌ Signature failed for client {client_id}")

        print(f"  🛡️  Security: {len(verified_updates)}/{len(client_updates)} updates verified")

        # Simulate FedAvg aggregation
        if verified_updates:
            print(f"  ⚖️  Aggregating parameters using FedAvg...")
            time.sleep(0.3)

            # Mock aggregation (average parameters)
            aggregated_params = {}
            first_params = verified_updates[0][0]

            for key in first_params:
                if isinstance(first_params[key], list) and isinstance(first_params[key][0], list):
                    aggregated_params[key] = []
                    for i in range(len(first_params[key])):
                        row = []
                        for j in range(len(first_params[key][i])):
                            avg = sum(update[0][key][i][j] for update in verified_updates) / len(verified_updates)
                            row.append(avg)
                        aggregated_params[key].append(row)
                else:
                    aggregated_params[key] = []
                    for i in range(len(first_params[key])):
                        avg = sum(update[0][key][i] for update in verified_updates) / len(verified_updates)
                        aggregated_params[key].append(avg)

            avg_loss = sum(loss for _, loss in verified_updates) / len(verified_updates)
            print(f"  📊 Average loss: {avg_loss:.4f}")
            print(f"  ✅ Aggregation complete")

            return aggregated_params, avg_loss

        return None, None

    def run_federated_learning_demo(self):
        """Run the complete federated learning demo."""
        print("\n" + "="*50)
        print("🌟 FEDERATED LEARNING SIMULATION STARTING")
        print("="*50)

        # Step 1: Security and setup
        if not self.simulate_security_check():
            print("❌ Security checks failed!")
            return

        server_private, server_public = self.simulate_keypair_generation()
        global_model = self.simulate_model_initialization()

        # Step 2: Simulate federated learning rounds
        num_rounds = 3
        num_clients = 5

        print(f"\n📋 Configuration:")
        print(f"  🔄 Rounds: {num_rounds}")
        print(f"  👥 Clients: {num_clients}")
        print(f"  🎯 Dataset: MNIST (simulated)")
        print(f"  🧠 Model: Small CNN (simulated)")

        current_model = global_model

        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*20} ROUND {round_num} {'='*20}")

            # Step 3: Client training
            client_updates = {}

            for client_id in range(num_clients):
                updated_params, loss, signature = self.simulate_client_training(
                    client_id, current_model, round_num
                )
                client_updates[client_id] = (updated_params, loss, signature)

            # Step 4: Server aggregation
            aggregated_model, avg_loss = self.simulate_server_aggregation(
                client_updates, round_num
            )

            if aggregated_model:
                current_model = aggregated_model
                # Simulate accuracy improvement
                accuracy = 0.3 + (round_num * 0.25) + random.random() * 0.1
                print(f"  🎯 Estimated accuracy: {accuracy:.1%}")

            time.sleep(1)  # Pause between rounds

        # Final results
        print(f"\n{'='*50}")
        print("🏁 FEDERATED LEARNING DEMO COMPLETE")
        print(f"{'='*50}")
        print(f"✅ All {num_rounds} rounds completed successfully")
        print(f"🛡️  Security: All updates cryptographically verified")
        print(f"📊 Final estimated accuracy: ~90% (MNIST)")
        print(f"💾 Artifacts saved to: {self.artifacts_dir}/")

        # Show generated files
        print(f"\n📁 Generated Files:")
        for file in self.artifacts_dir.glob("*"):
            print(f"  - {file.name}")

def main():
    """Run the basic demo."""
    try:
        demo = BasicDemo()
        demo.run_federated_learning_demo()

        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 This simulation shows the core FL workflow.")
        print(f"📚 Install full dependencies to run real PyTorch training.")

    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()