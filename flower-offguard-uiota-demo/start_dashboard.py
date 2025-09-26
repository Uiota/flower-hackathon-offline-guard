#!/usr/bin/env python3
"""
Quick launcher for the FL Dashboard
"""

import subprocess
import time
import os
import webbrowser

def find_free_port():
    """Find a free port to use."""
    import socket
    ports = [8081, 8082, 8083, 8084, 8085]

    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return 8081

def main():
    port = find_free_port()

    print("🚀 Launching State-of-the-Art Federated Learning Dashboard")
    print("=" * 60)
    print()
    print("✨ Features:")
    print("   🎨 Modern glass morphism design")
    print("   📊 Real-time training charts")
    print("   🔴 Live client monitoring")
    print("   💫 Animated particle background")
    print("   📱 Responsive mobile design")
    print()
    print(f"🌐 Dashboard URL: http://localhost:{port}")
    print("🎯 Click 'Start Training' to begin the demo")
    print()
    print("⏹️  Press Ctrl+C to stop")
    print()

    # Set environment
    os.environ["OFFLINE_MODE"] = "1"

    # Update dashboard server to use the free port
    with open('dashboard_server.py', 'r') as f:
        content = f.read()

    updated_content = content.replace(
        'def __init__(self, port=8080):',
        f'def __init__(self, port={port}):'
    )

    with open('dashboard_server.py', 'w') as f:
        f.write(updated_content)

    try:
        # Start server
        print("🔧 Starting dashboard server...")
        process = subprocess.Popen(['python3', 'dashboard_server.py'])

        # Wait a moment for server to start
        time.sleep(2)

        # Try to open browser
        try:
            webbrowser.open(f'http://localhost:{port}')
            print(f"🌐 Browser opened to http://localhost:{port}")
        except:
            print(f"🌐 Open your browser to: http://localhost:{port}")

        print("✅ Dashboard server running!")
        print()

        # Wait for process
        process.wait()

    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()