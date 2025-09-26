#!/usr/bin/env python3
"""
Website Generator Agent - Creates download website for the Flower Off-Guard UIOTA demo.

This agent handles:
- Static HTML website generation for demo download
- Demo description and installation guide
- Download buttons for different platforms
- API documentation extraction from code
- Integration with existing frontend structure
"""

import argparse
import ast
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🌐 [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class WebsiteGeneratorAgent:
    """Main website generator automation agent."""

    def __init__(self, project_root: Path, output_dir: Path = None, dist_dir: Path = None):
        self.project_root = Path(project_root).resolve()
        self.demo_dir = self.project_root / "flower-offguard-uiota-demo"
        self.frontend_dir = self.project_root / "frontend"
        self.dist_dir = Path(dist_dir) if dist_dir else self.project_root / "dist"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "website"

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Website configuration
        self.config = {
            "site_title": "Flower Off-Guard UIOTA Demo",
            "site_description": "Federated Learning with Security and Mesh Networking",
            "version": "1.0.0",
            "github_url": "https://github.com/uiota/offline-guard",
            "demo_features": [
                "🌸 Flower AI Integration",
                "🛡️ Off-Guard Security",
                "🌐 UIOTA Mesh Networking",
                "🔒 Differential Privacy",
                "🚀 CPU-Optimized",
                "📱 Cross-Platform"
            ],
            "supported_platforms": ["Linux", "macOS", "Windows"],
            "python_requirements": "Python 3.8+"
        }

    def extract_api_documentation(self) -> Dict:
        """Extract API documentation from source code."""
        logger.info("Extracting API documentation from source code...")

        api_docs = {
            "modules": [],
            "classes": [],
            "functions": []
        }

        src_dir = self.demo_dir / "src"
        if not src_dir.exists():
            logger.warning("Source directory not found, skipping API extraction")
            return api_docs

        for py_file in src_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                module_docs = self._parse_python_file(py_file)
                api_docs["modules"].append(module_docs)

            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")

        logger.info(f"✅ Extracted documentation for {len(api_docs['modules'])} modules")
        return api_docs

    def _parse_python_file(self, file_path: Path) -> Dict:
        """Parse Python file for documentation."""
        content = file_path.read_text(encoding='utf-8')

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return {"name": file_path.stem, "docstring": "", "classes": [], "functions": []}

        module_doc = {
            "name": file_path.stem,
            "docstring": ast.get_docstring(tree) or "",
            "classes": [],
            "functions": []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "methods": []
                }

                for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                    method_doc = {
                        "name": method.name,
                        "docstring": ast.get_docstring(method) or "",
                        "args": [arg.arg for arg in method.args.args]
                    }
                    class_doc["methods"].append(method_doc)

                module_doc["classes"].append(class_doc)

            elif isinstance(node, ast.FunctionDef) and not any(
                isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                if hasattr(parent, 'body') and node in getattr(parent, 'body', [])
            ):
                func_doc = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "args": [arg.arg for arg in node.args.args]
                }
                module_doc["functions"].append(func_doc)

        return module_doc

    def generate_html_template(self) -> str:
        """Generate base HTML template."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <meta name="description" content="{{description}}">

    <!-- Favicon -->
    <link rel="icon" type="image/png" href="assets/favicon.png">

    <!-- CSS -->
    <link rel="stylesheet" href="assets/style.css">

    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="nav-logo">
                <i class="fas fa-shield-alt"></i>
                <span>{{site_title}}</span>
            </div>
            <ul class="nav-menu">
                <li><a href="#home">Home</a></li>
                <li><a href="#features">Features</a></li>
                <li><a href="#download">Download</a></li>
                <li><a href="#docs">Documentation</a></li>
                <li><a href="#api">API</a></li>
            </ul>
        </div>
    </nav>

    <main>
        {{content}}
    </main>

    <footer class="footer">
        <div class="footer-container">
            <p>&copy; 2024 UIOTA Team. MIT License.</p>
            <div class="footer-links">
                <a href="{{github_url}}" target="_blank"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    </footer>

    <!-- JavaScript -->
    <script src="assets/script.js"></script>
</body>
</html>"""

    def generate_home_section(self) -> str:
        """Generate home section HTML."""
        features_html = "\\n".join([
            f'<li><span class="feature-icon">{feature.split()[0]}</span> {" ".join(feature.split()[1:])}</li>'
            for feature in self.config["demo_features"]
        ])

        return f"""
    <section id="home" class="hero">
        <div class="hero-container">
            <div class="hero-content">
                <h1 class="hero-title">
                    <span class="highlight">Flower Off-Guard</span> UIOTA Demo
                </h1>
                <p class="hero-subtitle">
                    Federated Learning with Advanced Security and Mesh Networking
                </p>
                <p class="hero-description">
                    Experience cutting-edge federated learning with built-in security features,
                    differential privacy, and peer-to-peer mesh networking - all optimized for
                    CPU-only environments.
                </p>

                <div class="hero-features">
                    <ul class="features-list">
                        {features_html}
                    </ul>
                </div>

                <div class="hero-actions">
                    <a href="#download" class="btn btn-primary">
                        <i class="fas fa-download"></i> Download Demo
                    </a>
                    <a href="#docs" class="btn btn-secondary">
                        <i class="fas fa-book"></i> Documentation
                    </a>
                </div>
            </div>

            <div class="hero-visual">
                <div class="demo-preview">
                    <div class="terminal-window">
                        <div class="terminal-header">
                            <span class="terminal-dot red"></span>
                            <span class="terminal-dot yellow"></span>
                            <span class="terminal-dot green"></span>
                        </div>
                        <div class="terminal-body">
                            <div class="terminal-line">$ python src/server.py --rounds 5</div>
                            <div class="terminal-line">🌸 Starting Flower server...</div>
                            <div class="terminal-line">🛡️ Off-Guard security initialized</div>
                            <div class="terminal-line">🌐 UIOTA mesh network active</div>
                            <div class="terminal-line">✅ Ready for federated learning!</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
"""

    def generate_features_section(self) -> str:
        """Generate features section HTML."""
        return """
    <section id="features" class="features">
        <div class="section-container">
            <h2 class="section-title">Why Choose Our Demo?</h2>

            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🌸</div>
                    <h3>Flower AI Integration</h3>
                    <p>Built on the robust Flower federated learning framework with custom strategies and security enhancements.</p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🛡️</div>
                    <h3>Off-Guard Security</h3>
                    <p>Comprehensive security framework with cryptographic protection, secure aggregation, and threat detection.</p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🌐</div>
                    <h3>UIOTA Mesh Network</h3>
                    <p>Peer-to-peer mesh networking for resilient, decentralized federated learning without central infrastructure.</p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3>Differential Privacy</h3>
                    <p>Optional differential privacy with configurable noise parameters to protect individual data privacy.</p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🚀</div>
                    <h3>CPU Optimized</h3>
                    <p>Designed for CPU-only environments - no GPU required. Perfect for edge devices and resource-constrained setups.</p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <h3>Cross-Platform</h3>
                    <p>Works seamlessly on Linux, macOS, and Windows with consistent behavior across all platforms.</p>
                </div>
            </div>
        </div>
    </section>
"""

    def generate_download_section(self) -> str:
        """Generate download section HTML."""
        platforms_html = ""
        for platform in self.config["supported_platforms"]:
            platform_lower = platform.lower()
            icon_map = {"linux": "fab fa-linux", "macos": "fab fa-apple", "windows": "fab fa-windows"}
            icon = icon_map.get(platform_lower, "fas fa-desktop")

            platforms_html += f"""
                <div class="download-card">
                    <div class="download-icon">
                        <i class="{icon}"></i>
                    </div>
                    <h3>{platform}</h3>
                    <p>Compatible with {platform} systems</p>
                    <div class="download-options">
                        <a href="downloads/flower-offguard-uiota-demo-1.0.0.tar.gz" class="btn btn-primary">
                            <i class="fas fa-download"></i> TAR.GZ
                        </a>
                        <a href="downloads/flower-offguard-uiota-demo-1.0.0.zip" class="btn btn-secondary">
                            <i class="fas fa-download"></i> ZIP
                        </a>
                    </div>
                </div>"""

        return f"""
    <section id="download" class="download">
        <div class="section-container">
            <h2 class="section-title">Download Demo</h2>
            <p class="section-description">
                Choose your platform and download the complete demo package with all dependencies and examples.
            </p>

            <div class="download-grid">
                {platforms_html}
            </div>

            <div class="download-info">
                <div class="info-card">
                    <h4><i class="fas fa-info-circle"></i> Requirements</h4>
                    <ul>
                        <li>{self.config["python_requirements"]}</li>
                        <li>2GB RAM minimum</li>
                        <li>500MB disk space</li>
                        <li>Internet connection (initial setup)</li>
                    </ul>
                </div>

                <div class="info-card">
                    <h4><i class="fas fa-rocket"></i> Quick Start</h4>
                    <ol>
                        <li>Download and extract package</li>
                        <li>Run <code>./install.sh</code> (Linux/Mac) or <code>install.bat</code> (Windows)</li>
                        <li>Start server: <code>python src/server.py</code></li>
                        <li>Start clients: <code>python src/client.py</code></li>
                    </ol>
                </div>

                <div class="info-card">
                    <h4><i class="fas fa-shield-alt"></i> Verify Download</h4>
                    <p>Check file integrity with SHA256:</p>
                    <code class="checksum-link">
                        <a href="downloads/CHECKSUMS.txt">View checksums</a>
                    </code>
                </div>
            </div>
        </div>
    </section>
"""

    def generate_docs_section(self) -> str:
        """Generate documentation section HTML."""
        return """
    <section id="docs" class="docs">
        <div class="section-container">
            <h2 class="section-title">Documentation</h2>

            <div class="docs-grid">
                <div class="doc-card">
                    <div class="doc-icon">
                        <i class="fas fa-play"></i>
                    </div>
                    <h3>Getting Started</h3>
                    <p>Quick setup guide and first steps with the demo.</p>
                    <a href="#getting-started" class="btn btn-outline">Read Guide</a>
                </div>

                <div class="doc-card">
                    <div class="doc-icon">
                        <i class="fas fa-cog"></i>
                    </div>
                    <h3>Configuration</h3>
                    <p>Customize federated learning parameters and security settings.</p>
                    <a href="#configuration" class="btn btn-outline">Learn More</a>
                </div>

                <div class="doc-card">
                    <div class="doc-icon">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <h3>Security Model</h3>
                    <p>Understand the Off-Guard security framework and features.</p>
                    <a href="#security" class="btn btn-outline">Security Docs</a>
                </div>

                <div class="doc-card">
                    <div class="doc-icon">
                        <i class="fas fa-network-wired"></i>
                    </div>
                    <h3>Mesh Networking</h3>
                    <p>UIOTA mesh networking setup and troubleshooting.</p>
                    <a href="#networking" class="btn btn-outline">Network Guide</a>
                </div>
            </div>
        </div>
    </section>
"""

    def generate_api_section(self, api_docs: Dict) -> str:
        """Generate API documentation section HTML."""
        modules_html = ""

        for module in api_docs.get("modules", []):
            classes_html = ""
            for cls in module.get("classes", []):
                methods_html = ""
                for method in cls.get("methods", []):
                    methods_html += f"""
                    <div class="method">
                        <h5>{method['name']}({', '.join(method['args'])})</h5>
                        <p>{method['docstring'] or 'No description available.'}</p>
                    </div>"""

                classes_html += f"""
                <div class="class">
                    <h4>class {cls['name']}</h4>
                    <p>{cls['docstring'] or 'No description available.'}</p>
                    <div class="methods">
                        {methods_html}
                    </div>
                </div>"""

            functions_html = ""
            for func in module.get("functions", []):
                functions_html += f"""
                <div class="function">
                    <h4>{func['name']}({', '.join(func['args'])})</h4>
                    <p>{func['docstring'] or 'No description available.'}</p>
                </div>"""

            modules_html += f"""
            <div class="module">
                <h3>{module['name']}.py</h3>
                <p>{module['docstring'] or 'No description available.'}</p>
                <div class="module-content">
                    {classes_html}
                    {functions_html}
                </div>
            </div>"""

        return f"""
    <section id="api" class="api">
        <div class="section-container">
            <h2 class="section-title">API Reference</h2>
            <p class="section-description">
                Comprehensive API documentation extracted from source code.
            </p>

            <div class="api-modules">
                {modules_html}
            </div>
        </div>
    </section>
"""

    def generate_css(self) -> str:
        """Generate CSS stylesheet."""
        return """/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    line-height: 1.6;
    color: #333;
    background: #fff;
}

/* Navigation */
.navbar {
    position: fixed;
    top: 0;
    width: 100%;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    z-index: 1000;
    padding: 1rem 0;
    border-bottom: 1px solid #e0e0e0;
}

.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 2rem;
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: bold;
    font-size: 1.25rem;
    color: #2563eb;
}

.nav-menu {
    display: flex;
    list-style: none;
    gap: 2rem;
}

.nav-menu a {
    text-decoration: none;
    color: #666;
    font-weight: 500;
    transition: color 0.3s;
}

.nav-menu a:hover {
    color: #2563eb;
}

/* Main content */
main {
    margin-top: 80px;
}

/* Hero section */
.hero {
    padding: 4rem 0 6rem;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}

.hero-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    align-items: center;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 1rem;
    line-height: 1.2;
}

.highlight {
    color: #2563eb;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: #666;
    margin-bottom: 1.5rem;
}

.hero-description {
    color: #555;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}

.features-list {
    list-style: none;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0.5rem;
    margin-bottom: 2rem;
}

.features-list li {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #555;
}

.feature-icon {
    font-size: 1.2rem;
}

.hero-actions {
    display: flex;
    gap: 1rem;
}

/* Terminal preview */
.demo-preview {
    background: #1e293b;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.terminal-window {
    background: #1e293b;
}

.terminal-header {
    background: #334155;
    padding: 0.75rem 1rem;
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

.terminal-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.terminal-dot.red { background: #ef4444; }
.terminal-dot.yellow { background: #f59e0b; }
.terminal-dot.green { background: #10b981; }

.terminal-body {
    padding: 1.5rem;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9rem;
}

.terminal-line {
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

.terminal-line:first-child {
    color: #60a5fa;
}

/* Sections */
.section-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 4rem 2rem;
}

.section-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
}

.section-description {
    text-align: center;
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 3rem;
}

/* Features grid */
.features {
    background: #f8fafc;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.feature-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    transition: transform 0.3s, box-shadow 0.3s;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.feature-card .feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.feature-card h3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #1e293b;
}

/* Download section */
.download-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
}

.download-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    border: 2px solid #e2e8f0;
    transition: border-color 0.3s;
}

.download-card:hover {
    border-color: #2563eb;
}

.download-icon {
    font-size: 3rem;
    color: #2563eb;
    margin-bottom: 1rem;
}

.download-options {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1rem;
}

.download-info {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 3rem;
}

.info-card {
    background: #f8fafc;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #2563eb;
}

.info-card h4 {
    margin-bottom: 1rem;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-card ul, .info-card ol {
    margin-left: 1.5rem;
}

.info-card code {
    background: #e2e8f0;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-family: 'Monaco', 'Menlo', monospace;
}

/* Documentation grid */
.docs {
    background: #f8fafc;
}

.docs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}

.doc-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.doc-icon {
    font-size: 2.5rem;
    color: #2563eb;
    margin-bottom: 1rem;
}

/* API section */
.api-modules {
    max-width: none;
}

.module {
    background: white;
    margin-bottom: 2rem;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

.module h3 {
    background: #f8fafc;
    padding: 1rem 1.5rem;
    margin: 0;
    border-bottom: 1px solid #e2e8f0;
    font-family: 'Monaco', 'Menlo', monospace;
}

.module-content {
    padding: 1.5rem;
}

.class, .function {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #f1f5f9;
}

.class h4, .function h4 {
    font-family: 'Monaco', 'Menlo', monospace;
    color: #2563eb;
    margin-bottom: 0.5rem;
}

.methods {
    margin-top: 1rem;
    margin-left: 1rem;
}

.method {
    margin-bottom: 1rem;
}

.method h5 {
    font-family: 'Monaco', 'Menlo', monospace;
    color: #059669;
    margin-bottom: 0.25rem;
}

/* Buttons */
.btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.95rem;
}

.btn-primary {
    background: #2563eb;
    color: white;
}

.btn-primary:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
}

.btn-secondary {
    background: #64748b;
    color: white;
}

.btn-secondary:hover {
    background: #475569;
}

.btn-outline {
    background: transparent;
    color: #2563eb;
    border: 2px solid #2563eb;
}

.btn-outline:hover {
    background: #2563eb;
    color: white;
}

/* Footer */
.footer {
    background: #1e293b;
    color: white;
    padding: 2rem 0;
    text-align: center;
}

.footer-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-links {
    display: flex;
    gap: 1rem;
}

.footer-links a {
    color: #94a3b8;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: color 0.3s;
}

.footer-links a:hover {
    color: white;
}

/* Responsive design */
@media (max-width: 768px) {
    .hero-container {
        grid-template-columns: 1fr;
        gap: 2rem;
        text-align: center;
    }

    .hero-title {
        font-size: 2rem;
    }

    .nav-container {
        flex-direction: column;
        gap: 1rem;
    }

    .nav-menu {
        gap: 1rem;
    }

    .footer-container {
        flex-direction: column;
        gap: 1rem;
    }

    .download-options {
        flex-direction: column;
    }
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Code styling */
pre {
    background: #1e293b;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    font-family: 'Monaco', 'Menlo', monospace;
    margin: 1rem 0;
}

code {
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-container > * {
    animation: fadeIn 0.6s ease-out;
}
"""

    def generate_javascript(self) -> str:
        """Generate JavaScript for website functionality."""
        return """// Website functionality
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add active state to navigation
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-menu a');

    function highlightNavigation() {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;

            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href').substring(1) === current) {
                item.classList.add('active');
            }
        });
    }

    // Highlight navigation on scroll
    window.addEventListener('scroll', highlightNavigation);

    // Terminal animation
    const terminalLines = document.querySelectorAll('.terminal-line');
    let lineIndex = 0;

    function typeNextLine() {
        if (lineIndex < terminalLines.length) {
            const line = terminalLines[lineIndex];
            line.style.opacity = '0';
            line.style.transform = 'translateX(-20px)';

            setTimeout(() => {
                line.style.transition = 'all 0.5s ease-out';
                line.style.opacity = '1';
                line.style.transform = 'translateX(0)';
                lineIndex++;

                setTimeout(typeNextLine, 800);
            }, 200);
        } else {
            // Restart animation after delay
            setTimeout(() => {
                lineIndex = 0;
                terminalLines.forEach(line => {
                    line.style.opacity = '0';
                    line.style.transform = 'translateX(-20px)';
                });
                setTimeout(typeNextLine, 1000);
            }, 5000);
        }
    }

    // Start terminal animation after page load
    setTimeout(typeNextLine, 2000);

    // Copy to clipboard functionality for checksums
    const checksumLinks = document.querySelectorAll('.checksum-link');
    checksumLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            // This would typically fetch and display checksum data
            showNotification('Checksum information copied to clipboard!');
        });
    });

    // Download tracking (optional analytics)
    const downloadButtons = document.querySelectorAll('a[href*="downloads/"]');
    downloadButtons.forEach(button => {
        button.addEventListener('click', function() {
            const fileName = this.getAttribute('href').split('/').pop();
            console.log(`Download started: ${fileName}`);
            showNotification(`Starting download: ${fileName}`);
        });
    });

    // Show notification function
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 1rem;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            z-index: 10000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease-out;
        `;

        document.body.appendChild(notification);

        // Show notification
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Hide notification
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Add API section toggle functionality
    const moduleHeaders = document.querySelectorAll('.module h3');
    moduleHeaders.forEach(header => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', function() {
            const moduleContent = this.nextElementSibling;
            const isHidden = moduleContent.style.display === 'none';

            moduleContent.style.display = isHidden ? 'block' : 'none';
            this.style.opacity = isHidden ? '1' : '0.7';
        });
    });

    // Initialize collapsed state for API modules
    document.querySelectorAll('.module-content').forEach(content => {
        content.style.display = 'none';
    });

    document.querySelectorAll('.module h3').forEach(header => {
        header.style.opacity = '0.7';
        header.title = 'Click to expand/collapse';
    });
});

// Add CSS for active navigation
const style = document.createElement('style');
style.textContent = `
    .nav-menu a.active {
        color: #2563eb;
        font-weight: 600;
    }

    .notification {
        font-weight: 500;
        font-size: 0.9rem;
    }
`;
document.head.appendChild(style);
"""

    def create_assets_directory(self) -> Path:
        """Create assets directory with required files."""
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Create CSS file
        (assets_dir / "style.css").write_text(self.generate_css())

        # Create JavaScript file
        (assets_dir / "script.js").write_text(self.generate_javascript())

        # Create favicon (simple SVG)
        favicon_svg = '''<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
    <rect width="32" height="32" rx="6" fill="#2563eb"/>
    <path d="M8 12h16v2H8v-2zM8 16h12v2H8v-2zM8 20h14v2H8v-2z" fill="white"/>
    <circle cx="24" cy="8" r="3" fill="#10b981"/>
</svg>'''
        (assets_dir / "favicon.svg").write_text(favicon_svg)

        return assets_dir

    def create_downloads_directory(self) -> Path:
        """Create downloads directory and copy distribution files."""
        downloads_dir = self.output_dir / "downloads"
        downloads_dir.mkdir(exist_ok=True)

        # Copy distribution files if they exist
        if self.dist_dir.exists():
            for dist_file in self.dist_dir.glob("*.tar.gz"):
                shutil.copy2(dist_file, downloads_dir)
            for dist_file in self.dist_dir.glob("*.zip"):
                shutil.copy2(dist_file, downloads_dir)

            # Copy checksums file
            checksums_file = self.dist_dir / "CHECKSUMS.txt"
            if checksums_file.exists():
                shutil.copy2(checksums_file, downloads_dir)

        return downloads_dir

    def generate_website(self) -> Dict:
        """Generate complete website."""
        logger.info("🚀 Starting website generation...")

        results = {
            "success": False,
            "website_path": self.output_dir,
            "files_created": [],
            "errors": []
        }

        try:
            # Extract API documentation
            api_docs = self.extract_api_documentation()

            # Generate HTML template
            template = self.generate_html_template()

            # Generate sections
            home_section = self.generate_home_section()
            features_section = self.generate_features_section()
            download_section = self.generate_download_section()
            docs_section = self.generate_docs_section()
            api_section = self.generate_api_section(api_docs)

            # Combine all sections
            content = home_section + features_section + download_section + docs_section + api_section

            # Replace template variables
            html_content = template.replace("{{title}}", f"{self.config['site_title']} - Download")
            html_content = html_content.replace("{{description}}", self.config["site_description"])
            html_content = html_content.replace("{{site_title}}", self.config["site_title"])
            html_content = html_content.replace("{{github_url}}", self.config["github_url"])
            html_content = html_content.replace("{{content}}", content)

            # Write main HTML file
            index_file = self.output_dir / "index.html"
            index_file.write_text(html_content)
            results["files_created"].append(str(index_file))

            # Create assets directory
            assets_dir = self.create_assets_directory()
            results["files_created"].extend([
                str(assets_dir / "style.css"),
                str(assets_dir / "script.js"),
                str(assets_dir / "favicon.svg")
            ])

            # Create downloads directory
            downloads_dir = self.create_downloads_directory()
            results["files_created"].append(str(downloads_dir))

            # Generate additional pages
            self._generate_additional_pages(api_docs)

            results["success"] = True
            logger.info("🎉 Website generation completed successfully!")

        except Exception as e:
            logger.error(f"Website generation failed: {e}")
            results["errors"].append(str(e))

        return results

    def _generate_additional_pages(self, api_docs: Dict) -> None:
        """Generate additional standalone pages."""

        # Generate getting started page
        getting_started = f"""
        <div class="section-container">
            <h1>Getting Started</h1>
            <div class="content">
                <h2>Installation</h2>
                <ol>
                    <li>Download the demo package for your platform</li>
                    <li>Extract the archive</li>
                    <li>Run the installation script</li>
                </ol>

                <h3>Linux/macOS</h3>
                <pre><code>tar -xzf flower-offguard-uiota-demo-1.0.0.tar.gz
cd flower-offguard-uiota-demo
./install.sh</code></pre>

                <h3>Windows</h3>
                <pre><code># Extract ZIP file
cd flower-offguard-uiota-demo
install.bat</code></pre>

                <h2>Running the Demo</h2>
                <h3>Start the Server</h3>
                <pre><code>python src/server.py --rounds 5 --clients-per-round 10</code></pre>

                <h3>Start Clients</h3>
                <pre><code>python src/client.py --server-address localhost:8080</code></pre>

                <h2>Configuration</h2>
                <p>Edit <code>config/demo.yaml</code> to customize settings:</p>
                <ul>
                    <li>Dataset selection (MNIST, CIFAR-10)</li>
                    <li>Federated learning strategy</li>
                    <li>Security parameters</li>
                    <li>Network simulation settings</li>
                </ul>
            </div>
        </div>
        """

        # Save additional pages if needed
        # This is a simplified version - in practice you'd create full HTML pages

    def validate_website(self) -> bool:
        """Validate generated website structure."""
        logger.info("Validating generated website...")

        required_files = [
            self.output_dir / "index.html",
            self.output_dir / "assets" / "style.css",
            self.output_dir / "assets" / "script.js",
            self.output_dir / "assets" / "favicon.svg",
        ]

        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Missing required file: {file_path}")
                return False

        logger.info("✅ Website validation passed")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Flower Off-Guard UIOTA Website Generator")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for website")
    parser.add_argument("--dist-dir", help="Distribution files directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize generator
    generator = WebsiteGeneratorAgent(
        project_root=args.project_root,
        output_dir=args.output_dir,
        dist_dir=args.dist_dir
    )

    # Generate website
    results = generator.generate_website()

    # Print results
    if results["success"]:
        logger.info("🌐 Website Generation Summary:")
        logger.info(f"   Website path: {results['website_path']}")
        logger.info(f"   Files created: {len(results['files_created'])}")
        for file_path in results["files_created"]:
            logger.info(f"   - {file_path}")

        # Validate website
        if generator.validate_website():
            logger.info("✅ Website validation passed")
            return 0
        else:
            logger.error("❌ Website validation failed")
            return 1
    else:
        logger.error("❌ Website generation failed!")
        for error in results["errors"]:
            logger.error(f"   - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())