"""
Functional P2P Mesh Network for Offline Federated Learning
"""

import asyncio
import json
import logging
import socket
import threading
import time
import uuid
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import qrcode
from io import BytesIO
import base64
import sys

sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    generate_keypair, sign_data, verify_signature,
    bytes_to_base64, base64_to_bytes, save_json, load_json
)

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a peer in the network."""
    peer_id: str
    address: str
    port: int
    public_key: bytes
    last_seen: float
    status: str = "online"
    capabilities: List[str] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class Message:
    """Network message structure."""
    message_id: str
    sender_id: str
    recipient_id: str  # "broadcast" for broadcast messages
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    signature: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create from dictionary."""
        return cls(**data)

    def sign(self, private_key: bytes) -> None:
        """Sign the message."""
        # Create message hash without signature
        msg_dict = self.to_dict()
        msg_dict.pop("signature", None)
        msg_bytes = json.dumps(msg_dict, sort_keys=True).encode()
        signature = sign_data(private_key, msg_bytes)
        self.signature = bytes_to_base64(signature)

    def verify(self, public_key: bytes) -> bool:
        """Verify message signature."""
        if not self.signature:
            return False

        try:
            # Recreate message hash without signature
            msg_dict = self.to_dict()
            msg_dict.pop("signature", None)
            msg_bytes = json.dumps(msg_dict, sort_keys=True).encode()
            signature = base64_to_bytes(self.signature)
            return verify_signature(public_key, signature, msg_bytes)
        except Exception as e:
            logger.warning(f"Message verification failed: {e}")
            return False


class P2PNode:
    """P2P network node with mesh capabilities."""

    def __init__(self, node_id: str = None, port: int = 0):
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.is_running = False

        # Security
        self.private_key, self.public_key = generate_keypair()

        # Network state
        self.peers: Dict[str, PeerInfo] = {}
        self.connections: Dict[str, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.seen_messages: Set[str] = set()

        # Server components
        self.server = None
        self.server_task = None

        # Event loop for async operations
        self.loop = None
        self.network_thread = None

        # Message queues
        self.outbound_messages = asyncio.Queue()
        self.inbound_messages = asyncio.Queue()

        # Default message handlers
        self._register_default_handlers()

        logger.info(f"P2P Node {self.node_id} initialized")

    def _register_default_handlers(self):
        """Register default message handlers."""
        self.register_handler("peer_discovery", self._handle_peer_discovery)
        self.register_handler("peer_announcement", self._handle_peer_announcement)
        self.register_handler("heartbeat", self._handle_heartbeat)
        self.register_handler("data_sync", self._handle_data_sync)

    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")

    async def start(self, host: str = "0.0.0.0", discovery_peers: List[str] = None):
        """Start the P2P node."""
        if self.is_running:
            logger.warning("Node is already running")
            return

        try:
            # Start server
            self.server = await asyncio.start_server(
                self._handle_client,
                host,
                self.port
            )

            # Get the actual port if 0 was specified
            if self.port == 0:
                self.port = self.server.sockets[0].getsockname()[1]

            self.is_running = True

            logger.info(f"P2P Node {self.node_id} started on {host}:{self.port}")

            # Start background tasks
            asyncio.create_task(self._heartbeat_task())
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._message_processor())

            # Discover initial peers
            if discovery_peers:
                await self._discover_peers(discovery_peers)

            # Start serving
            await self.server.serve_forever()

        except Exception as e:
            logger.error(f"Failed to start P2P node: {e}")
            self.is_running = False

    def start_threaded(self, host: str = "0.0.0.0", discovery_peers: List[str] = None):
        """Start the P2P node in a separate thread."""
        if self.network_thread and self.network_thread.is_alive():
            logger.warning("Network thread is already running")
            return

        def run_network():
            """Run the network in its own event loop."""
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self.start(host, discovery_peers))
            except Exception as e:
                logger.error(f"Network thread error: {e}")

        self.network_thread = threading.Thread(target=run_network, daemon=True)
        self.network_thread.start()

        # Wait for startup
        time.sleep(1)

    async def stop(self):
        """Stop the P2P node."""
        if not self.is_running:
            return

        self.is_running = False

        # Close all connections
        for peer_id, (reader, writer) in self.connections.items():
            writer.close()
            await writer.wait_closed()

        self.connections.clear()

        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info(f"P2P Node {self.node_id} stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming client connections."""
        client_address = writer.get_extra_info('peername')
        logger.debug(f"New connection from {client_address}")

        try:
            while True:
                # Read message length
                length_data = await reader.readexactly(4)
                message_length = int.from_bytes(length_data, byteorder='big')

                # Read message data
                message_data = await reader.readexactly(message_length)
                message_json = message_data.decode('utf-8')

                # Parse and handle message
                try:
                    message_dict = json.loads(message_json)
                    message = Message.from_dict(message_dict)
                    await self._handle_message(message, reader, writer)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from {client_address}: {e}")

        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by {client_address}")
        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_message(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming messages."""
        # Check if we've seen this message before (prevent loops)
        if message.message_id in self.seen_messages:
            return

        self.seen_messages.add(message.message_id)

        # Verify message signature if sender is known
        if message.sender_id in self.peers:
            peer = self.peers[message.sender_id]
            if not message.verify(peer.public_key):
                logger.warning(f"Invalid signature from {message.sender_id}")
                return

        # Handle the message
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message, reader, writer)
            except Exception as e:
                logger.error(f"Error handling message {message.message_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")

        # Forward broadcast messages to other peers
        if message.recipient_id == "broadcast" and message.sender_id != self.node_id:
            await self._forward_message(message)

    async def _forward_message(self, message: Message):
        """Forward a message to other peers."""
        for peer_id, (reader, writer) in self.connections.items():
            if peer_id != message.sender_id:
                try:
                    await self._send_message_to_writer(message, writer)
                except Exception as e:
                    logger.warning(f"Failed to forward message to {peer_id}: {e}")

    async def send_message(self, recipient_id: str, message_type: str, payload: Dict[str, Any]):
        """Send a message to a specific peer or broadcast."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=time.time()
        )

        # Sign the message
        message.sign(self.private_key)

        if recipient_id == "broadcast":
            await self._broadcast_message(message)
        else:
            await self._send_to_peer(recipient_id, message)

    async def _broadcast_message(self, message: Message):
        """Broadcast a message to all connected peers."""
        failed_peers = []

        for peer_id, (reader, writer) in self.connections.items():
            try:
                await self._send_message_to_writer(message, writer)
            except Exception as e:
                logger.warning(f"Failed to send to {peer_id}: {e}")
                failed_peers.append(peer_id)

        # Remove failed connections
        for peer_id in failed_peers:
            await self._remove_peer(peer_id)

    async def _send_to_peer(self, peer_id: str, message: Message):
        """Send a message to a specific peer."""
        if peer_id in self.connections:
            reader, writer = self.connections[peer_id]
            try:
                await self._send_message_to_writer(message, writer)
            except Exception as e:
                logger.warning(f"Failed to send to {peer_id}: {e}")
                await self._remove_peer(peer_id)
        else:
            logger.warning(f"No connection to peer {peer_id}")

    async def _send_message_to_writer(self, message: Message, writer: asyncio.StreamWriter):
        """Send a message through a writer."""
        message_json = json.dumps(message.to_dict())
        message_data = message_json.encode('utf-8')
        message_length = len(message_data)

        # Send length followed by data
        writer.write(message_length.to_bytes(4, byteorder='big'))
        writer.write(message_data)
        await writer.drain()

    async def connect_to_peer(self, address: str, port: int) -> bool:
        """Connect to a peer."""
        try:
            reader, writer = await asyncio.open_connection(address, port)

            # Send peer announcement
            await self.send_message("broadcast", "peer_announcement", {
                "peer_id": self.node_id,
                "address": address,
                "port": self.port,
                "public_key": bytes_to_base64(self.public_key),
                "capabilities": ["fl_client", "mesh_node"]
            })

            logger.info(f"Connected to peer at {address}:{port}")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to {address}:{port}: {e}")
            return False

    async def _discover_peers(self, discovery_peers: List[str]):
        """Discover peers from a list of addresses."""
        for peer_address in discovery_peers:
            try:
                if ":" in peer_address:
                    address, port = peer_address.split(":")
                    port = int(port)
                else:
                    address = peer_address
                    port = 8081  # Default port

                await self.connect_to_peer(address, port)
                await asyncio.sleep(1)  # Stagger connections
            except Exception as e:
                logger.warning(f"Failed to discover peer {peer_address}: {e}")

    async def _handle_peer_discovery(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle peer discovery messages."""
        # Respond with our peer information
        await self.send_message(message.sender_id, "peer_announcement", {
            "peer_id": self.node_id,
            "address": "localhost",  # In production, use actual address
            "port": self.port,
            "public_key": bytes_to_base64(self.public_key),
            "capabilities": ["fl_client", "mesh_node"]
        })

    async def _handle_peer_announcement(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle peer announcement messages."""
        payload = message.payload
        peer_id = payload.get("peer_id")

        if peer_id and peer_id != self.node_id:
            # Add or update peer
            public_key = base64_to_bytes(payload.get("public_key", ""))
            peer = PeerInfo(
                peer_id=peer_id,
                address=payload.get("address", ""),
                port=payload.get("port", 0),
                public_key=public_key,
                last_seen=time.time(),
                status="online",
                capabilities=payload.get("capabilities", [])
            )

            self.peers[peer_id] = peer
            self.connections[peer_id] = (reader, writer)

            logger.info(f"Added peer {peer_id} from {peer.address}:{peer.port}")

    async def _handle_heartbeat(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle heartbeat messages."""
        if message.sender_id in self.peers:
            self.peers[message.sender_id].last_seen = time.time()
            self.peers[message.sender_id].status = "online"

    async def _handle_data_sync(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle data synchronization messages."""
        # This would be implemented based on specific sync requirements
        logger.info(f"Data sync request from {message.sender_id}: {message.payload}")

    async def _heartbeat_task(self):
        """Send periodic heartbeats."""
        while self.is_running:
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
            if self.is_running:
                await self.send_message("broadcast", "heartbeat", {
                    "timestamp": time.time(),
                    "status": "online"
                })

    async def _cleanup_task(self):
        """Clean up inactive peers."""
        while self.is_running:
            await asyncio.sleep(60)  # Cleanup every minute
            if self.is_running:
                current_time = time.time()
                inactive_peers = []

                for peer_id, peer in self.peers.items():
                    if current_time - peer.last_seen > 120:  # 2 minutes timeout
                        inactive_peers.append(peer_id)

                for peer_id in inactive_peers:
                    await self._remove_peer(peer_id)

    async def _message_processor(self):
        """Process outbound messages."""
        while self.is_running:
            try:
                # Process any queued messages
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Message processor error: {e}")

    async def _remove_peer(self, peer_id: str):
        """Remove a peer from the network."""
        if peer_id in self.peers:
            del self.peers[peer_id]

        if peer_id in self.connections:
            reader, writer = self.connections[peer_id]
            writer.close()
            await writer.wait_closed()
            del self.connections[peer_id]

        logger.info(f"Removed peer {peer_id}")

    def get_peer_count(self) -> int:
        """Get the number of connected peers."""
        return len(self.peers)

    def get_peers(self) -> List[PeerInfo]:
        """Get list of all peers."""
        return list(self.peers.values())

    def get_network_info(self) -> Dict:
        """Get network status information."""
        return {
            "node_id": self.node_id,
            "port": self.port,
            "is_running": self.is_running,
            "peer_count": self.get_peer_count(),
            "peers": [asdict(peer) for peer in self.get_peers()]
        }

    def generate_qr_code(self) -> str:
        """Generate QR code for peer discovery."""
        peer_info = {
            "node_id": self.node_id,
            "address": "localhost",  # In production, use actual address
            "port": self.port,
            "public_key": bytes_to_base64(self.public_key)
        }

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(peer_info))
        qr.make(fit=True)

        # Create QR code image
        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to base64 string
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"


class MeshNetworkManager:
    """High-level mesh network manager."""

    def __init__(self, node_id: str = None):
        self.node = P2PNode(node_id)
        self.data_store: Dict[str, Any] = {}
        self.sync_handlers: Dict[str, Callable] = {}

        # Register mesh-specific handlers
        self.node.register_handler("file_sync", self._handle_file_sync)
        self.node.register_handler("model_sync", self._handle_model_sync)

    def start(self, port: int = 0, discovery_peers: List[str] = None):
        """Start the mesh network."""
        self.node.start_threaded(port=port, discovery_peers=discovery_peers)

    def stop(self):
        """Stop the mesh network."""
        if self.node.loop:
            asyncio.run_coroutine_threadsafe(self.node.stop(), self.node.loop)

    def broadcast_data(self, data_type: str, data: Any):
        """Broadcast data to all peers."""
        if self.node.loop:
            asyncio.run_coroutine_threadsafe(
                self.node.send_message("broadcast", "data_sync", {
                    "data_type": data_type,
                    "data": data,
                    "timestamp": time.time()
                }),
                self.node.loop
            )

    def sync_file(self, file_path: str, file_data: bytes):
        """Synchronize a file across the network."""
        file_hash = hashlib.sha256(file_data).hexdigest()

        if self.node.loop:
            asyncio.run_coroutine_threadsafe(
                self.node.send_message("broadcast", "file_sync", {
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "file_data": bytes_to_base64(file_data),
                    "timestamp": time.time()
                }),
                self.node.loop
            )

    async def _handle_file_sync(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle file synchronization."""
        payload = message.payload
        file_path = payload.get("file_path")
        file_hash = payload.get("file_hash")
        file_data_b64 = payload.get("file_data")

        if file_path and file_data_b64:
            try:
                file_data = base64_to_bytes(file_data_b64)
                computed_hash = hashlib.sha256(file_data).hexdigest()

                if computed_hash == file_hash:
                    # Save file
                    full_path = Path(f"./mesh_data/{file_path}")
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(full_path, "wb") as f:
                        f.write(file_data)

                    logger.info(f"Synced file: {file_path}")
                else:
                    logger.warning(f"File hash mismatch for {file_path}")

            except Exception as e:
                logger.error(f"Error syncing file {file_path}: {e}")

    async def _handle_model_sync(self, message: Message, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle model synchronization."""
        payload = message.payload
        model_id = payload.get("model_id")
        model_data = payload.get("model_data")

        if model_id and model_data:
            self.data_store[f"model_{model_id}"] = model_data
            logger.info(f"Synced model: {model_id}")

    def get_status(self) -> Dict:
        """Get mesh network status."""
        return self.node.get_network_info()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P2P Mesh Network Node")
    parser.add_argument("--node-id", help="Node identifier")
    parser.add_argument("--port", type=int, default=8081, help="Listen port")
    parser.add_argument("--peers", nargs="*", help="Initial peers to connect to")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # Create and start mesh network
    mesh = MeshNetworkManager(args.node_id)
    mesh.start(args.port, args.peers)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down mesh network")
        mesh.stop()