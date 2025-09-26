# Deployment Automation Agents

Comprehensive deployment automation system for the Flower Off-Guard UIOTA demo, providing fully automated building, packaging, and website generation.

## 🎯 Overview

This deployment system consists of four specialized automation agents that work together to create a complete deployment pipeline:

1. **Demo Builder Agent** - Packages the demo with dependencies
2. **Website Generator Agent** - Creates download website
3. **SDK Packager Agent** - Creates pip-installable SDK
4. **Deployment Orchestrator Agent** - Coordinates the entire process

## 🚀 Quick Start

### Basic Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete deployment
python deployment-orchestrator-agent.py

# Or use the main entry point
python ../deploy-demo.py
```

### Interactive Mode

```bash
# Interactive deployment with options
python ../deploy-demo.py --interactive
```

## 📦 Individual Agents

Each agent can be run independently for specific tasks:

### Demo Builder Agent

Automates demo package creation with virtual environments, dependency installation, testing, and distribution packaging.

```bash
python demo-builder-agent.py --project-root .. --output-dir ../dist
```

**Features:**
- ✅ Virtual environment creation and management
- 📚 Dependency installation from requirements.txt
- 🧪 Test execution (if tests exist)
- 📦 Multiple package formats (tar.gz, zip)
- 🔧 Installation script generation
- 📋 Configuration file creation

**Outputs:**
- `flower-offguard-uiota-demo-1.0.0.tar.gz` - Linux/macOS package
- `flower-offguard-uiota-demo-1.0.0.zip` - Cross-platform package
- `CHECKSUMS.txt` - SHA256 checksums
- Installation scripts (`install.sh`, `install.bat`)

### Website Generator Agent

Creates a professional download website with API documentation extraction.

```bash
python website-generator-agent.py --project-root .. --output-dir ../website --dist-dir ../dist
```

**Features:**
- 🌐 Responsive HTML5 website
- 📖 Auto-generated API documentation
- 💾 Download buttons for all platforms
- 🎨 Modern CSS styling with animations
- 📱 Mobile-friendly design
- ⚡ JavaScript interactivity

**Outputs:**
- `index.html` - Main website page
- `assets/` - CSS, JavaScript, images
- API documentation sections
- Download integration

### SDK Packager Agent

Transforms the demo into a pip-installable Python SDK.

```bash
python sdk-packager-agent.py --project-root .. --output-dir ../sdk-dist
```

**Features:**
- 📦 Proper Python package structure
- 🔧 setuptools configuration
- 📝 Console script entry points
- 📚 Example usage scripts
- 🐍 Wheel and source distributions
- 🛠️ Development dependencies

**Outputs:**
- `flower_offguard_uiota-1.0.0-py3-none-any.whl` - Wheel package
- `flower-offguard-uiota-1.0.0.tar.gz` - Source distribution
- Example scripts and documentation

### Deployment Orchestrator Agent

Coordinates the entire deployment pipeline with error handling and progress tracking.

```bash
python deployment-orchestrator-agent.py --project-root .. --output-dir ../deployment-output
```

**Features:**
- 🎯 Complete pipeline orchestration
- 🔄 Retry logic with exponential backoff
- 📊 Progress tracking and reporting
- 🗂️ Artifact collection and organization
- 🐳 Podman container integration
- 📋 Comprehensive deployment reports

## 🛠️ Configuration

### Environment Variables

```bash
export DEPLOYMENT_OUTPUT_DIR="/path/to/output"
export DEPLOYMENT_SKIP_TESTS="false"
export DEPLOYMENT_CLEANUP="true"
```

### Agent Configuration

Each agent supports configuration through command-line arguments:

```bash
# Common options
--project-root /path/to/project    # Project root directory
--output-dir /path/to/output       # Output directory
--verbose                          # Enable verbose logging
--help                             # Show help message

# Demo Builder specific
--skip-tests                       # Skip test execution
--no-cleanup                       # Keep build artifacts

# Website Generator specific
--dist-dir /path/to/distributions  # Distribution files location

# SDK Packager specific
--skip-validation                  # Skip package validation

# Orchestrator specific
--resume                           # Resume previous deployment
--skip-stages demo-builder         # Skip specific stages
```

## 📁 Output Structure

The deployment system creates a well-organized output structure:

```
deployment-output/
├── builds/                 # Temporary build artifacts
├── distributions/          # Demo packages
│   ├── flower-offguard-uiota-demo-1.0.0.tar.gz
│   ├── flower-offguard-uiota-demo-1.0.0.zip
│   └── CHECKSUMS.txt
├── sdk/                    # SDK packages
│   ├── flower_offguard_uiota-1.0.0-py3-none-any.whl
│   └── flower-offguard-uiota-1.0.0.tar.gz
├── website/                # Generated website
│   ├── index.html
│   ├── assets/
│   └── downloads/
├── logs/                   # Deployment logs
│   └── deployment-20241122-143022.log
├── deployment-state.json  # Execution state
└── DEPLOYMENT_REPORT.md   # Final report
```

## 🔧 Integration

### Podman Container Integration

The deployment system integrates with the existing Podman infrastructure:

```bash
# Automatic integration during deployment
python deployment-orchestrator-agent.py

# Manual integration
cp sdk-dist/*.whl ../containers/ml-toolkit/
podman build -t offline-guard-ml-updated -f ../containers/ml-toolkit/Containerfile ..
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Deploy Demo
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r deployment-agents/requirements.txt
      - name: Run deployment
        run: python deploy-demo.py --non-interactive
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: deployment-artifacts
          path: deployment-output/
```

### Custom Scripting

```python
#!/usr/bin/env python3
"""Custom deployment script example."""

from pathlib import Path
from deployment_agents.deployment_orchestrator_agent import DeploymentOrchestratorAgent

def custom_deployment():
    """Run custom deployment with specific configuration."""
    orchestrator = DeploymentOrchestratorAgent(
        project_root=Path.cwd(),
        output_dir=Path("custom-output")
    )

    # Customize configuration
    orchestrator.config.update({
        "cleanup_on_success": False,
        "retry_attempts": 5,
        "integration_tests": True
    })

    # Run deployment
    results = orchestrator.run_deployment(
        skip_stages=["website-generator"]  # Skip website for faster deployment
    )

    return results["success"]

if __name__ == "__main__":
    success = custom_deployment()
    exit(0 if success else 1)
```

## 🧪 Testing

### Running Agent Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific agent tests
pytest tests/test_demo_builder.py -v
pytest tests/test_website_generator.py -v
pytest tests/test_sdk_packager.py -v
pytest tests/test_orchestrator.py -v
```

### Manual Testing

```bash
# Test demo builder
python demo-builder-agent.py --project-root .. --output-dir test-output

# Test website generator
python website-generator-agent.py --project-root .. --output-dir test-website

# Test SDK packager
python sdk-packager-agent.py --project-root .. --output-dir test-sdk

# Test full orchestration
python deployment-orchestrator-agent.py --project-root .. --output-dir test-deployment
```

## 🐛 Troubleshooting

### Common Issues

#### Environment Validation Failures

```bash
# Check Python version
python --version  # Should be 3.8+

# Check project structure
ls ../flower-offguard-uiota-demo/src/  # Should contain Python files

# Check dependencies
pip install -r requirements.txt
```

#### Build Failures

```bash
# Check demo requirements
ls ../flower-offguard-uiota-demo/requirements.txt

# Test demo manually
cd ../flower-offguard-uiota-demo
python -m venv test-env
source test-env/bin/activate  # Linux/macOS
# or test-env\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Website Generation Issues

```bash
# Check for API documentation extraction
python -c "import ast; print('AST parsing works')"

# Test website generation manually
python website-generator-agent.py --verbose --project-root ..
```

#### SDK Packaging Problems

```bash
# Check setuptools version
pip install --upgrade setuptools wheel

# Test SDK building manually
cd test-sdk-build
python setup.py sdist bdist_wheel
```

### Debug Mode

```bash
# Enable maximum verbosity
python deployment-orchestrator-agent.py --verbose

# Check logs
tail -f deployment-output/logs/deployment-*.log

# Resume failed deployment
python deployment-orchestrator-agent.py --resume
```

### Performance Optimization

```bash
# Skip tests for faster builds
python demo-builder-agent.py --skip-tests

# Skip optional stages
python deployment-orchestrator-agent.py --skip-stages website-generator

# Use parallel processing (where supported)
export PYTHONPATH=.
python -m concurrent.futures
```

## 📋 Requirements

### System Requirements

- Python 3.8+
- 2GB RAM minimum
- 1GB free disk space
- Internet connection (for dependencies)

### Optional Requirements

- Podman (for container integration)
- Git (for version control integration)
- Node.js (for advanced website features)

### Platform Support

- ✅ Linux (Ubuntu 20.04+, CentOS 8+)
- ✅ macOS (10.15+)
- ✅ Windows (10+)

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/uiota/offline-guard.git
cd offline-guard/deployment-agents

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Linux/macOS
# or dev-env\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Code Style

```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
mypy *.py
```

### Adding New Agents

1. Create new agent script: `my-agent.py`
2. Follow the existing agent pattern
3. Add to orchestrator pipeline
4. Update documentation
5. Add tests

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

## 📞 Support

- 📚 Documentation: https://github.com/uiota/offline-guard
- 🐛 Issues: https://github.com/uiota/offline-guard/issues
- 📧 Email: dev@uiota.org

---

Made with ❤️ by the UIOTA Team