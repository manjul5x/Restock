#!/usr/bin/env python3
"""
Run script for the demand visualization web application.
"""

import sys
import socket
from pathlib import Path

# Add the parent directory to the path so we can import the forecaster package
sys.path.append(str(Path(__file__).parent.parent))

from webapp.app import app

def find_available_port(start_port=8080, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    print("Starting Demand Visualization Web Application...")
    
    # Find available port
    port = find_available_port(8080)
    if port is None:
        print("❌ Error: No available ports found in range 8080-8089")
        sys.exit(1)
    
    print(f"Open your browser and go to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Try a different port or check if another application is using the port") 