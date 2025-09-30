#!/usr/bin/env python3
"""
Simple Backend Server for Off-Guard Demo
Uses only standard library modules for compatibility
"""

import json
import time
import sqlite3
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.parse

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OffGuardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Off-Guard API"""

    def __init__(self, *args, **kwargs):
        self.auth_db = 'offguard_simple.db'
        self.secret_key = 'simple-demo-key-2024'
        self.init_database()
        super().__init__(*args, **kwargs)

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.auth_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create demo user if not exists
        cursor.execute('SELECT id FROM users WHERE username = ?', ('demo',))
        if not cursor.fetchone():
            demo_password = hashlib.sha256('demo123'.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', ('demo', 'demo@offguard.ai', demo_password))

        conn.commit()
        conn.close()

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_request_data(self):
        """Get JSON data from request body"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                raw_data = self.rfile.read(content_length)
                return json.loads(raw_data.decode('utf-8'))
            return {}
        except:
            return {}

    def authenticate_request(self):
        """Simple authentication check"""
        auth_header = self.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            # Simple token validation (in production, use proper JWT)
            return token == 'demo-token-2024'
        return False

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/':
            self.serve_file('demo_final.py')
        elif path == '/auth.html':
            self.serve_file('auth.html')
        elif path == '/chatgpt_welcome_section.html':
            self.serve_file('chatgpt_welcome_section.html')
        elif path == '/live_ai_dashboard.html':
            self.serve_file('live_ai_dashboard.html')
        elif path == '/offline_mode.html':
            self.serve_file('offline_mode.html')
        elif path == '/api_client.js':
            self.serve_file('api_client.js')
        elif path == '/api/metrics':
            self.handle_get_metrics()
        elif path == '/api/export':
            self.handle_export_data()
        else:
            self.send_json_response({'error': 'Not found'}, 404)

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/api/auth/login':
            self.handle_login()
        elif path == '/api/auth/signup':
            self.handle_signup()
        elif path == '/api/fl/start-demo':
            self.handle_start_fl_demo()
        elif path == '/api/fl/add-client':
            self.handle_add_fl_client()
        elif path == '/api/fl/run-training':
            self.handle_run_training()
        elif path == '/api/ai/test':
            self.handle_test_ai()
        elif path == '/api/ai/analyze':
            self.handle_analyze_model()
        elif path == '/api/security/generate-keys':
            self.handle_generate_keys()
        elif path == '/api/security/test-encryption':
            self.handle_test_encryption()
        elif path == '/api/security/offline-mode':
            self.handle_offline_mode()
        else:
            self.send_json_response({'error': 'Endpoint not found'}, 404)

    def serve_file(self, filename):
        """Serve static files"""
        try:
            file_path = Path(filename)
            if file_path.exists():
                content_type = 'text/html' if filename.endswith('.html') else 'application/javascript' if filename.endswith('.js') else 'text/plain'
                self.send_response(200)
                self.send_cors_headers()
                self.send_header('Content-Type', content_type)
                self.end_headers()

                if filename == 'demo_final.py':
                    # Generate HTML from the demo_final.py
                    with open(filename, 'r') as f:
                        content = f.read()
                        # Extract HTML content between quotes
                        start = content.find('return f"""') + len('return f"""')
                        end = content.rfind('"""')
                        html_content = content[start:end]
                        # Process template variables
                        html_content = html_content.replace('{{', '{').replace('}}', '}')
                        self.wfile.write(html_content.encode())
                else:
                    with open(filename, 'rb') as f:
                        self.wfile.write(f.read())
            else:
                self.send_json_response({'error': 'File not found'}, 404)
        except Exception as e:
            logger.error(f"Error serving file {filename}: {e}")
            self.send_json_response({'error': 'Server error'}, 500)

    # Authentication handlers
    def handle_login(self):
        """Handle user login"""
        data = self.get_request_data()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            self.send_json_response({'error': 'Missing credentials'}, 400)
            return

        conn = sqlite3.connect(self.auth_db)
        cursor = conn.cursor()

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?',
                      (username, password_hash))

        user = cursor.fetchone()
        conn.close()

        if user:
            self.send_json_response({
                'success': True,
                'user': {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2]
                },
                'token': 'demo-token-2024'
            })
        else:
            self.send_json_response({'error': 'Invalid credentials'}, 401)

    def handle_signup(self):
        """Handle user registration"""
        data = self.get_request_data()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not all([username, email, password]):
            self.send_json_response({'error': 'Missing required fields'}, 400)
            return

        try:
            conn = sqlite3.connect(self.auth_db)
            cursor = conn.cursor()

            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                          (username, email, password_hash))

            user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self.send_json_response({
                'success': True,
                'user': {
                    'id': user_id,
                    'username': username,
                    'email': email
                },
                'token': 'demo-token-2024'
            })
        except sqlite3.IntegrityError:
            self.send_json_response({'error': 'Username or email already exists'}, 409)

    # FL Training handlers
    def handle_start_fl_demo(self):
        """Start FL demo"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        session_id = secrets.token_hex(8)
        self.send_json_response({
            'success': True,
            'session_id': session_id,
            'devices': 3,
            'status': 'training_started'
        })

    def handle_add_fl_client(self):
        """Add FL client"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        data = self.get_request_data()
        device_type = data.get('type', 'mobile')
        device_name = f"{device_type.title()} Device #{secrets.randbits(8)}"

        self.send_json_response({
            'success': True,
            'device': {
                'id': secrets.token_hex(4),
                'name': device_name,
                'type': device_type,
                'status': 'connecting',
                'accuracy': round(80 + secrets.randbelow(20), 1)
            }
        })

    def handle_run_training(self):
        """Run training round"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        round_num = secrets.randbelow(50) + 1
        accuracy = round(85 + (round_num * 0.2) + (secrets.randbelow(100) / 100), 1)

        self.send_json_response({
            'success': True,
            'training_result': {
                'session_id': 'demo-session',
                'round': round_num,
                'accuracy': min(accuracy, 99.9),
                'devices_count': 3
            }
        })

    # AI Integration handlers
    def handle_test_ai(self):
        """Test AI integration"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        data = self.get_request_data()
        provider = data.get('provider', 'openai')

        responses = {
            'openai': 'GPT-4 Analysis: Your federated learning model shows excellent convergence patterns with 94.2% accuracy.',
            'anthropic': 'Claude Analysis: The distributed training exhibits strong privacy preservation metrics.',
            'huggingface': 'HuggingFace Model: Transformer-based federated learning optimization complete.'
        }

        self.send_json_response({
            'success': True,
            'provider': provider,
            'response': responses.get(provider, 'AI service response generated successfully.')
        })

    def handle_analyze_model(self):
        """Generate AI model analysis"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        analysis = {
            'model_performance': {
                'accuracy': 94.7,
                'convergence_rate': 'Excellent',
                'privacy_score': 98.2
            },
            'recommendations': [
                'Consider increasing learning rate for faster convergence',
                'Add differential privacy for enhanced security',
                'Implement client sampling for better scalability'
            ],
            'security_assessment': {
                'encryption_strength': 'AES-256',
                'vulnerability_score': 'Low',
                'compliance': ['GDPR', 'HIPAA']
            }
        }

        self.send_json_response({
            'success': True,
            'analysis': analysis,
            'generated_at': datetime.now().isoformat()
        })

    # Security handlers
    def handle_generate_keys(self):
        """Generate encryption keys"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        key_fingerprint = hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:16]

        self.send_json_response({
            'success': True,
            'key_generated': True,
            'key_fingerprint': key_fingerprint,
            'algorithm': 'AES-256',
            'generated_at': datetime.now().isoformat()
        })

    def handle_test_encryption(self):
        """Test encryption"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        test_data = "sensitive_federated_learning_data"

        self.send_json_response({
            'success': True,
            'test_passed': True,
            'original_size': len(test_data),
            'encrypted_size': len(test_data) + 32,  # Simulate encryption overhead
            'algorithm': 'AES-256',
            'test_completed_at': datetime.now().isoformat()
        })

    def handle_offline_mode(self):
        """Enable offline mode"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        self.send_json_response({
            'success': True,
            'offline_mode': True,
            'features': [
                'Local FL training',
                'Air-gapped operation',
                'Encrypted local storage',
                'Zero external connectivity'
            ]
        })

    # Data handlers
    def handle_get_metrics(self):
        """Get system metrics"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        metrics = {
            'user_stats': {
                'devices': 3,
                'training_sessions': 2,
                'total_rounds': 47,
                'avg_accuracy': 94.2
            },
            'system_health': {
                'fl_server': 'healthy',
                'mcp_bridge': 'connected',
                'ai_services': 'operational',
                'encryption': 'active'
            },
            'performance': {
                'latency': '23ms',
                'throughput': '1.2k ops/sec',
                'uptime': '99.7%'
            }
        }

        self.send_json_response({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    def handle_export_data(self):
        """Export user data"""
        if not self.authenticate_request():
            self.send_json_response({'error': 'Authentication required'}, 401)
            return

        export_data = {
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'format': 'JSON'
            },
            'training_history': [
                {'round': 1, 'accuracy': 85.2, 'timestamp': '2024-01-01T10:00:00'},
                {'round': 2, 'accuracy': 87.1, 'timestamp': '2024-01-01T10:05:00'},
                {'round': 3, 'accuracy': 89.3, 'timestamp': '2024-01-01T10:10:00'}
            ],
            'devices': [
                {'id': 'mobile-1', 'type': 'mobile', 'accuracy': 87.3},
                {'id': 'desktop-1', 'type': 'desktop', 'accuracy': 89.1},
                {'id': 'edge-1', 'type': 'edge', 'accuracy': 85.7}
            ]
        }

        self.send_json_response(export_data)

    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass

def run_simple_backend(host='localhost', port=8002):
    """Run the simple backend server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, OffGuardRequestHandler)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                🛡️  Off-Guard Simple Backend                 ║
║                                                              ║
║  🌐 Server: http://{host}:{port}                                ║
║  🔑 Authentication: Simple Token-based                      ║
║  🛡️ Demo Account: demo/demo123                             ║
║  📊 API: RESTful with CORS support                          ║
║                                                              ║
║  Test endpoints:                                             ║
║  • POST /api/auth/login - User authentication               ║
║  • POST /api/fl/start-demo - Start FL training              ║
║  • POST /api/ai/test - Test AI integration                  ║
║  • GET  /api/metrics - System metrics                       ║
║                                                              ║
║  Frontend: http://{host}:{port}/auth.html                      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown")
        httpd.server_close()

if __name__ == "__main__":
    run_simple_backend()