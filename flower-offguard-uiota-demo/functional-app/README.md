# Functional Federated Learning Application

A complete, production-ready federated learning system with real ML training, web dashboard, mesh networking, and containerized deployment.

## Features

### 🧠 **Real Machine Learning**
- **Actual model training** with PyTorch (CNN, Linear models)
- **Real datasets** (MNIST, CIFAR-10, synthetic data)
- **Non-IID data partitioning** using Dirichlet distribution
- **Differential privacy** support with Opacus
- **Real federated aggregation** (FedAvg and custom strategies)

### 🌐 **Web Dashboard**
- **Real-time training visualization** with live charts
- **Interactive controls** to start/stop training
- **Client monitoring** with status and metrics
- **WebSocket connections** for live updates
- **Responsive design** with Bootstrap UI

### 🔗 **Mesh Networking**
- **P2P communication** between clients
- **Offline capabilities** with data synchronization
- **QR code verification** for peer discovery
- **Cryptographic signing** and message verification
- **Network topology management**

### 📊 **Live Monitoring**
- **System resource monitoring** (CPU, memory, network)
- **Training progress tracking** with metrics collection
- **Client status dashboard** with real-time updates
- **Network topology visualization**

### 🐳 **Containerized Deployment**
- **Docker containers** for all components
- **Docker Compose** orchestration
- **Service discovery** and load balancing
- **Persistent data storage** with volumes
- **Health checks** and auto-restart

## Quick Start

### 1. Simple Demo (Local)

```bash
# Install dependencies
pip install -r containers/requirements.txt

# Run the complete demo
python run_demo.py --dataset mnist --num-clients 3 --num-rounds 5

# Access dashboard at http://localhost:5000
```

### 2. Manual Start (Components)

```bash
# Terminal 1: Start FL Server
python run_server.py --dataset mnist --num-rounds 10

# Terminal 2: Start Client 1
python run_client.py --client-id client-1 --dataset mnist

# Terminal 3: Start Client 2
python run_client.py --client-id client-2 --dataset mnist

# Terminal 4: Start Client 3 with mesh networking
python run_client.py --client-id client-3 --dataset mnist --enable-mesh
```

### 3. Docker Deployment

```bash
# Build and start all services
cd containers
docker-compose up --build

# Scale clients
docker-compose up --scale fl-client-1=3

# View logs
docker-compose logs -f fl-server

# Access dashboard at http://localhost:80
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │   FL Server     │    │   FL Clients    │
│   (Flask/React) │◄──►│   (Flower)      │◄──►│   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └──────────────►│  Live Monitor   │◄────────────┘
                        │  (Metrics)      │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │  Mesh Network   │
                        │  (P2P/Offline)  │
                        └─────────────────┘
```

## Components

### 📂 **Directory Structure**

```
functional-app/
├── shared/           # Shared modules
│   ├── models.py     # ML models (CNN, Linear)
│   ├── datasets.py   # Data handling & partitioning
│   └── utils.py      # Utilities & cryptography
├── app/              # Web application
│   ├── server.py     # Flask server with FL integration
│   ├── templates/    # HTML templates
│   └── static/       # CSS/JS assets
├── client/           # FL client
│   └── fl_client.py  # Functional FL client
├── mesh/             # Mesh networking
│   └── p2p_network.py # P2P network implementation
├── dashboard/        # Monitoring
│   └── monitor.py    # Real-time monitoring
├── containers/       # Docker deployment
│   ├── Dockerfile.*  # Container definitions
│   ├── docker-compose.yml
│   └── requirements.txt
└── config/           # Configuration files
```

### 🔧 **Configuration Options**

#### Dataset & Model
- **Datasets**: MNIST, CIFAR-10, Synthetic
- **Models**: CNN (Convolutional), Linear
- **Data distribution**: Configurable non-IID with Dirichlet

#### Training Parameters
- **Local epochs**: 1-10 per client per round
- **Batch size**: 16-128
- **Learning rate**: 0.001-0.1
- **FL rounds**: 1-100

#### Security & Privacy
- **Cryptographic signing**: RSA-2048 with message verification
- **Differential privacy**: Configurable ε, δ parameters
- **Secure aggregation**: Parameter signing and verification

## API Reference

### REST Endpoints

```bash
# Server status
GET /api/status

# Start/stop training
POST /api/start
POST /api/stop

# Get metrics
GET /api/metrics

# Configuration
GET /api/config
POST /api/config
```

### WebSocket Events

```javascript
// Client events
socket.emit('request_status');
socket.emit('request_metrics');

// Server events
socket.on('status_update', data => { ... });
socket.on('metrics_update', data => { ... });
socket.on('training_started', data => { ... });
socket.on('training_stopped', data => { ... });
```

## Advanced Usage

### Custom Datasets

```python
# Add custom dataset to shared/datasets.py
def get_custom_client_data(client_id, num_clients, **kwargs):
    # Implement custom data loading
    return train_loader, test_loader
```

### Custom Models

```python
# Add custom model to shared/models.py
class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define model architecture

    def forward(self, x):
        # Implement forward pass
        return x
```

### Custom FL Strategy

```python
# Extend the FL strategy in app/server.py
class CustomStrategy(CustomFedAvgStrategy):
    def aggregate_fit(self, server_round, results, failures):
        # Implement custom aggregation
        return super().aggregate_fit(server_round, results, failures)
```

## Deployment Modes

### 1. Development Mode
```bash
python run_demo.py --log-level DEBUG
```

### 2. Production Mode
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### 3. Offline Mode
```bash
# Start with mesh networking enabled
python run_client.py --client-id client-1 --enable-mesh --mesh-peers peer1:8081
```

### 4. Distributed Mode
```bash
# Server on machine 1
python run_server.py --host 0.0.0.0

# Clients on machine 2
python run_client.py --server-address 192.168.1.100:8080
```

## Monitoring & Debugging

### Real-time Metrics
- **Training loss/accuracy** per round
- **Client participation** and status
- **System resources** (CPU, memory, network)
- **Mesh network topology**

### Logging
```bash
# Enable debug logging
python run_demo.py --log-level DEBUG --log-file demo.log

# View Docker logs
docker-compose logs -f --tail=100 fl-server
```

### Health Checks
- **HTTP endpoints** for service status
- **WebSocket connectivity** monitoring
- **Client heartbeat** tracking
- **Automatic restart** on failure

## Security Features

### Cryptographic Protection
- **RSA-2048** key pairs for each participant
- **Message signing** for parameter updates
- **Signature verification** before aggregation
- **QR code verification** for peer discovery

### Privacy Protection
- **Differential privacy** with configurable budgets
- **Secure aggregation** protocols
- **Local data isolation** - no raw data sharing
- **Non-IID simulation** for realistic scenarios

## Performance Optimization

### Training Optimization
- **GPU acceleration** (CUDA/MPS support)
- **Batch processing** for efficiency
- **Asynchronous operations** with proper threading
- **Memory management** for large models

### Network Optimization
- **WebSocket compression** for real-time data
- **Efficient serialization** of model parameters
- **Connection pooling** for multiple clients
- **Load balancing** with nginx

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Change default ports
   python run_server.py --web-port 5001 --fl-port 8081
   ```

2. **Memory issues with large models**
   ```bash
   # Reduce batch size
   python run_client.py --batch-size 16
   ```

3. **Docker connectivity**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect functional-app_fl_network
   ```

4. **Client connection failures**
   ```bash
   # Check server logs
   docker-compose logs fl-server

   # Verify server is listening
   netstat -an | grep 8080
   ```

## Contributing

1. **Add new datasets** in `shared/datasets.py`
2. **Implement new models** in `shared/models.py`
3. **Extend FL strategies** in `app/server.py`
4. **Add monitoring metrics** in `dashboard/monitor.py`
5. **Improve mesh protocols** in `mesh/p2p_network.py`

## License

This functional federated learning application is provided for educational and research purposes. See the original project license for details.

---

**🚀 Ready to run real federated learning with actual ML training!**