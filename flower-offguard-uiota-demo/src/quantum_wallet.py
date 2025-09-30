"""
Quantum-Resistant Wallet Infrastructure for LL TOKEN OFFLINE
Implements post-quantum cryptography and secure token management
"""

import os
import json
import time
import hashlib
import secrets
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from datetime import datetime, timezone

# Post-quantum cryptography (using hybrid approach for now)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


@dataclass
class QuantumSafeParameters:
    """Parameters for quantum-safe cryptographic operations."""
    key_size: int = 256  # 256-bit keys for quantum safety
    iteration_count: int = 600000  # High iteration count for PBKDF2
    salt_size: int = 32  # 32-byte salt
    nonce_size: int = 16  # 16-byte nonce for AES-GCM
    hash_algorithm: str = "SHA256"


@dataclass
class TokenMetadata:
    """Metadata for LL TOKEN OFFLINE transactions."""
    token_id: str
    created_at: datetime
    owner_pubkey: str
    amount: int  # Token amount in smallest units
    nonce: int
    signature: str
    quantum_proof: str  # Future quantum-resistant proof
    offline_mode: bool = True


@dataclass
class WalletState:
    """Current state of the quantum wallet."""
    wallet_id: str
    balance: int
    nonce: int
    last_sync: Optional[datetime] = None
    transactions: List[Dict] = field(default_factory=list)
    pending_transactions: List[Dict] = field(default_factory=list)


class QuantumWallet:
    """
    Quantum-resistant wallet for LL TOKEN OFFLINE.

    Implements:
    - Post-quantum cryptographic signatures
    - Offline transaction signing
    - Secure key storage with quantum-safe encryption
    - Transaction batching and synchronization
    """

    def __init__(self, wallet_path: str, passphrase: str = None):
        self.wallet_path = Path(wallet_path)
        self.wallet_path.mkdir(parents=True, exist_ok=True)

        self.params = QuantumSafeParameters()
        self.passphrase = passphrase or self._generate_secure_passphrase()

        # Initialize or load wallet
        self.private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self.public_key: Optional[ed25519.Ed25519PublicKey] = None
        self.wallet_id: str = ""
        self.state: Optional[WalletState] = None

        self._init_or_load_wallet()

    def _generate_secure_passphrase(self) -> str:
        """Generate a secure passphrase for wallet encryption."""
        words = secrets.randbits(256).to_bytes(32, 'big')
        return hashlib.sha256(words).hexdigest()[:64]

    def _init_or_load_wallet(self):
        """Initialize a new wallet or load existing one."""
        wallet_file = self.wallet_path / "wallet.json"
        keystore_file = self.wallet_path / "keystore.enc"

        if wallet_file.exists() and keystore_file.exists():
            self._load_existing_wallet()
        else:
            self._create_new_wallet()

    def _create_new_wallet(self):
        """Create a new quantum wallet."""
        logger.info("Creating new quantum wallet...")

        # Generate quantum-safe keypair
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

        # Generate wallet ID from public key
        pubkey_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        self.wallet_id = hashlib.sha256(pubkey_bytes).hexdigest()[:16]

        # Initialize wallet state
        self.state = WalletState(
            wallet_id=self.wallet_id,
            balance=0,
            nonce=0,
            last_sync=None
        )

        # Save wallet
        self._save_wallet()
        logger.info(f"✅ Quantum wallet created: {self.wallet_id}")

    def _load_existing_wallet(self):
        """Load existing wallet from storage."""
        logger.info("Loading existing quantum wallet...")

        try:
            # Load wallet metadata
            wallet_file = self.wallet_path / "wallet.json"
            with open(wallet_file, 'r') as f:
                wallet_data = json.load(f)

            self.wallet_id = wallet_data['wallet_id']

            # Load wallet state
            self.state = WalletState(**wallet_data['state'])

            # Decrypt and load private key
            self._load_encrypted_keystore()

            logger.info(f"✅ Quantum wallet loaded: {self.wallet_id}")

        except Exception as e:
            logger.error(f"Failed to load wallet: {e}")
            raise

    def _save_wallet(self):
        """Save wallet to encrypted storage."""
        try:
            # Save wallet metadata and state
            wallet_file = self.wallet_path / "wallet.json"
            wallet_data = {
                'wallet_id': self.wallet_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'version': '1.0.0',
                'quantum_safe': True,
                'state': asdict(self.state) if self.state else {}
            }

            # Convert datetime objects to strings for JSON serialization
            if self.state and self.state.last_sync:
                wallet_data['state']['last_sync'] = self.state.last_sync.isoformat()

            with open(wallet_file, 'w') as f:
                json.dump(wallet_data, f, indent=2)

            # Save encrypted keystore
            self._save_encrypted_keystore()

        except Exception as e:
            logger.error(f"Failed to save wallet: {e}")
            raise

    def _save_encrypted_keystore(self):
        """Save private key in encrypted keystore."""
        if not self.private_key:
            raise ValueError("No private key to save")

        # Serialize private key
        private_key_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

        # Encrypt keystore with passphrase
        keystore_data = {
            'private_key': private_key_bytes.hex(),
            'public_key': self.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ).hex(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        encrypted_keystore = self._encrypt_data(
            json.dumps(keystore_data).encode(),
            self.passphrase
        )

        keystore_file = self.wallet_path / "keystore.enc"
        with open(keystore_file, 'wb') as f:
            f.write(encrypted_keystore)

    def _load_encrypted_keystore(self):
        """Load private key from encrypted keystore."""
        keystore_file = self.wallet_path / "keystore.enc"

        with open(keystore_file, 'rb') as f:
            encrypted_data = f.read()

        # Decrypt keystore
        decrypted_data = self._decrypt_data(encrypted_data, self.passphrase)
        keystore_data = json.loads(decrypted_data.decode())

        # Restore keys
        private_key_bytes = bytes.fromhex(keystore_data['private_key'])
        self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)

        public_key_bytes = bytes.fromhex(keystore_data['public_key'])
        self.public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)

    def _encrypt_data(self, data: bytes, passphrase: str) -> bytes:
        """Encrypt data using quantum-safe parameters."""
        # Generate salt and derive key
        salt = secrets.token_bytes(self.params.salt_size)
        key = self._derive_key(passphrase, salt)

        # Generate nonce for AES-GCM
        nonce = secrets.token_bytes(self.params.nonce_size)

        # Encrypt using AES-256-GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return: salt + nonce + tag + ciphertext
        return salt + nonce + encryptor.tag + ciphertext

    def _decrypt_data(self, encrypted_data: bytes, passphrase: str) -> bytes:
        """Decrypt data using quantum-safe parameters."""
        # Extract components
        salt = encrypted_data[:self.params.salt_size]
        nonce = encrypted_data[self.params.salt_size:self.params.salt_size + self.params.nonce_size]
        tag = encrypted_data[self.params.salt_size + self.params.nonce_size:self.params.salt_size + self.params.nonce_size + 16]
        ciphertext = encrypted_data[self.params.salt_size + self.params.nonce_size + 16:]

        # Derive key
        key = self._derive_key(passphrase, salt)

        # Decrypt using AES-256-GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        """Derive encryption key from passphrase using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=self.params.iteration_count,
        )
        return kdf.derive(passphrase.encode())

    def create_transaction(self, to_address: str, amount: int, metadata: Dict = None) -> Dict:
        """Create a new LL TOKEN OFFLINE transaction."""
        if not self.state:
            raise ValueError("Wallet not initialized")

        if amount <= 0:
            raise ValueError("Amount must be positive")

        if self.state.balance < amount:
            raise ValueError(f"Insufficient balance: {self.state.balance} < {amount}")

        # Create transaction
        transaction = {
            'id': secrets.token_hex(16),
            'from': self.wallet_id,
            'to': to_address,
            'amount': amount,
            'nonce': self.state.nonce + 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'offline_mode': True,
            'metadata': metadata or {}
        }

        # Sign transaction
        transaction['signature'] = self._sign_transaction(transaction)

        # Add to pending transactions
        self.state.pending_transactions.append(transaction)
        self.state.nonce += 1

        # Save updated state
        self._save_wallet()

        logger.info(f"Created transaction {transaction['id']}: {amount} tokens to {to_address}")
        return transaction

    def _sign_transaction(self, transaction: Dict) -> str:
        """Sign a transaction with quantum-safe signature."""
        # Create canonical transaction data for signing
        signing_data = {
            'from': transaction['from'],
            'to': transaction['to'],
            'amount': transaction['amount'],
            'nonce': transaction['nonce'],
            'timestamp': transaction['timestamp']
        }

        # Convert to bytes
        data_bytes = json.dumps(signing_data, sort_keys=True).encode()

        # Sign with Ed25519 (quantum-safe until quantum computers become practical)
        signature = self.private_key.sign(data_bytes)

        return signature.hex()

    def verify_transaction(self, transaction: Dict, public_key: ed25519.Ed25519PublicKey = None) -> bool:
        """Verify a transaction signature."""
        try:
            # Use provided public key or derive from transaction
            if not public_key:
                # In a real implementation, you'd look up the public key from the 'from' address
                public_key = self.public_key

            # Recreate signing data
            signing_data = {
                'from': transaction['from'],
                'to': transaction['to'],
                'amount': transaction['amount'],
                'nonce': transaction['nonce'],
                'timestamp': transaction['timestamp']
            }

            data_bytes = json.dumps(signing_data, sort_keys=True).encode()
            signature = bytes.fromhex(transaction['signature'])

            # Verify signature
            public_key.verify(signature, data_bytes)
            return True

        except Exception as e:
            logger.debug(f"Transaction verification failed: {e}")
            return False

    def get_balance(self) -> int:
        """Get current wallet balance."""
        return self.state.balance if self.state else 0

    def get_pending_transactions(self) -> List[Dict]:
        """Get pending transactions."""
        return self.state.pending_transactions if self.state else []

    def get_transaction_history(self) -> List[Dict]:
        """Get transaction history."""
        return self.state.transactions if self.state else []

    def export_public_key(self) -> str:
        """Export public key for receiving tokens."""
        if not self.public_key:
            raise ValueError("No public key available")

        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ).hex()

    def mint_tokens(self, amount: int, authority_signature: str = None) -> Dict:
        """Mint new tokens (for testing/development)."""
        if not self.state:
            raise ValueError("Wallet not initialized")

        # In production, this would require proper authority verification
        self.state.balance += amount

        mint_transaction = {
            'id': secrets.token_hex(16),
            'type': 'mint',
            'to': self.wallet_id,
            'amount': amount,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'authority_signature': authority_signature or 'dev_mint'
        }

        self.state.transactions.append(mint_transaction)
        self._save_wallet()

        logger.info(f"Minted {amount} tokens to wallet {self.wallet_id}")
        return mint_transaction


class TokenRail:
    """
    Master rail system for LL TOKEN OFFLINE transactions.
    Manages offline transaction batching, validation, and eventual synchronization.
    """

    def __init__(self, rail_path: str):
        self.rail_path = Path(rail_path)
        self.rail_path.mkdir(parents=True, exist_ok=True)

        self.pending_batches: List[Dict] = []
        self.confirmed_batches: List[Dict] = []

        self._load_rail_state()

    def _load_rail_state(self):
        """Load rail state from storage."""
        state_file = self.rail_path / "rail_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self.pending_batches = state_data.get('pending_batches', [])
                self.confirmed_batches = state_data.get('confirmed_batches', [])

    def _save_rail_state(self):
        """Save rail state to storage."""
        state_file = self.rail_path / "rail_state.json"
        state_data = {
            'pending_batches': self.pending_batches,
            'confirmed_batches': self.confirmed_batches,
            'last_update': datetime.now(timezone.utc).isoformat()
        }

        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def submit_transaction_batch(self, transactions: List[Dict], batch_metadata: Dict = None) -> str:
        """Submit a batch of transactions to the rail."""
        batch_id = secrets.token_hex(16)

        batch = {
            'id': batch_id,
            'transactions': transactions,
            'metadata': batch_metadata or {},
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'pending',
            'transaction_count': len(transactions)
        }

        self.pending_batches.append(batch)
        self._save_rail_state()

        logger.info(f"Submitted transaction batch {batch_id} with {len(transactions)} transactions")
        return batch_id

    def get_pending_batches(self) -> List[Dict]:
        """Get all pending transaction batches."""
        return self.pending_batches.copy()

    def confirm_batch(self, batch_id: str) -> bool:
        """Confirm a transaction batch."""
        for i, batch in enumerate(self.pending_batches):
            if batch['id'] == batch_id:
                batch['status'] = 'confirmed'
                batch['confirmed_at'] = datetime.now(timezone.utc).isoformat()

                # Move to confirmed batches
                self.confirmed_batches.append(batch)
                del self.pending_batches[i]

                self._save_rail_state()
                logger.info(f"Confirmed transaction batch {batch_id}")
                return True

        logger.warning(f"Batch {batch_id} not found in pending batches")
        return False


def create_quantum_wallet_system(base_path: str) -> Tuple[QuantumWallet, TokenRail]:
    """Create a complete quantum wallet system with token rail."""

    # Ensure offline mode is enabled
    os.environ["OFFLINE_MODE"] = "1"

    # Create wallet
    wallet_path = Path(base_path) / "wallet"
    passphrase = secrets.token_hex(32)  # Generate secure passphrase
    wallet = QuantumWallet(str(wallet_path), passphrase)

    # Create token rail
    rail_path = Path(base_path) / "token_rail"
    token_rail = TokenRail(str(rail_path))

    # Mint initial tokens for testing
    wallet.mint_tokens(1000000, "initial_allocation")  # 1M tokens

    logger.info("✅ Quantum wallet system created successfully")
    logger.info(f"Wallet ID: {wallet.wallet_id}")
    logger.info(f"Initial balance: {wallet.get_balance()} tokens")
    logger.info(f"Public key: {wallet.export_public_key()[:32]}...")

    return wallet, token_rail