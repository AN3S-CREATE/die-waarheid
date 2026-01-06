#!/usr/bin/env python3
"""
Die Waarheid - Unified Service Launcher

This script starts all required services for the Die Waarheid forensic analysis system:
- FastAPI backend server (port 8000)
- Streamlit web interface (port 8501)
- React frontend (port 5173) - if available

Usage:
    python die_waarheid/launcher.py
    
    Or from the root directory:
    python -m die_waarheid.launcher
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages all Die Waarheid services"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.shutdown_event = threading.Event()
        self.root_dir = Path(__file__).parent.parent
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        missing_deps = []
        
        # Check Python dependencies
        try:
            import streamlit
            import fastapi
            import uvicorn
            logger.info("✅ Python dependencies found")
        except ImportError as e:
            missing_deps.append(f"Python: {e}")
        
        # Check if Node.js is available for React frontend
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"✅ Node.js found: {result.stdout.strip()}")
            else:
                logger.warning("⚠️ Node.js not found - React frontend will not be available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Node.js not found - React frontend will not be available")
        
        if missing_deps:
            logger.error("❌ Missing dependencies:")
            for dep in missing_deps:
                logger.error(f"   - {dep}")
            logger.error("\n💡 To fix, run: pip install -r requirements.txt")
            return False
        
        return True
    
    def check_environment(self) -> bool:
        """Check environment configuration"""
        logger.info("🔧 Checking environment configuration...")
        
        env_file = self.root_dir / '.env'
        if not env_file.exists():
            logger.warning("⚠️ .env file not found")
            logger.info("💡 Copy .env.example to .env and configure your API keys")
            logger.info("   cp .env.example .env")
            return True  # Not critical for basic functionality
        
        # Load environment variables
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key and value and not value.startswith('your_'):
                            os.environ[key] = value
        except Exception as e:
            logger.warning(f"⚠️ Error reading .env file: {e}")
        
        # Check critical environment variables
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key or gemini_key.startswith('your_'):
            logger.warning("⚠️ GEMINI_API_KEY not configured - AI analysis will not work")
            logger.info("💡 Get your API key from: https://makersuite.google.com/app/apikey")
        
        api_key = os.getenv('API_KEY')
        if not api_key or api_key.startswith('generate_'):
            logger.warning("⚠️ API_KEY not configured - generating temporary key")
            import secrets
            temp_key = secrets.token_urlsafe(32)
            os.environ['API_KEY'] = temp_key
            logger.info(f"🔑 Temporary API key: {temp_key}")
        
        logger.info("✅ Environment check completed")
        return True
    
    def start_fastapi(self) -> Optional[subprocess.Popen]:
        """Start FastAPI backend server"""
        logger.info("🚀 Starting FastAPI backend server...")
        
        try:
            # Change to the correct directory
            api_module = "die_waarheid.api_server:app"
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                api_module,
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
                "--log-level", "info"
            ]
            
            proc = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(proc)
            logger.info("✅ FastAPI server started on http://localhost:8000")
            logger.info("📚 API docs available at http://localhost:8000/docs")
            
            return proc
            
        except Exception as e:
            logger.error(f"❌ Failed to start FastAPI server: {e}")
            return None
    
    def start_streamlit(self) -> Optional[subprocess.Popen]:
        """Start Streamlit web interface"""
        logger.info("🚀 Starting Streamlit web interface...")
        
        try:
            app_path = self.root_dir / "die_waarheid" / "app.py"
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
                "--browser.gatherUsageStats=false"
            ]
            
            proc = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(proc)
            logger.info("✅ Streamlit interface started on http://localhost:8501")
            
            return proc
            
        except Exception as e:
            logger.error(f"❌ Failed to start Streamlit: {e}")
            return None
    
    def start_react_frontend(self) -> Optional[subprocess.Popen]:
        """Start React frontend (if available)"""
        frontend_dir = self.root_dir / "frontend"
        
        if not frontend_dir.exists():
            logger.info("ℹ️ React frontend directory not found - skipping")
            return None
        
        package_json = frontend_dir / "package.json"
        if not package_json.exists():
            logger.info("ℹ️ React frontend not configured - skipping")
            return None
        
        # Check if node_modules exists
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            logger.warning("⚠️ React dependencies not installed")
            logger.info("💡 Run: cd frontend && npm install")
            return None
        
        logger.info("🚀 Starting React frontend...")
        
        try:
            cmd = ["npm", "run", "dev"]
            
            proc = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(proc)
            logger.info("✅ React frontend started on http://localhost:5173")
            
            return proc
            
        except Exception as e:
            logger.error(f"❌ Failed to start React frontend: {e}")
            return None
    
    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        
        return False
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while not self.shutdown_event.is_set():
            for i, proc in enumerate(self.processes[:]):  # Copy list to avoid modification during iteration
                if proc.poll() is not None:  # Process has terminated
                    logger.warning(f"⚠️ Process {i} has terminated unexpectedly")
                    self.processes.remove(proc)
            
            time.sleep(5)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"\n🛑 Received signal {signum}, shutting down...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_all(self):
        """Start all services"""
        logger.info("🚀 Starting Die Waarheid Forensic Analysis System")
        logger.info("=" * 60)
        
        # Pre-flight checks
        if not self.check_dependencies():
            sys.exit(1)
        
        if not self.check_environment():
            logger.warning("⚠️ Environment issues detected, but continuing...")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Start services
        logger.info("\n📡 Starting services...")
        
        # Start FastAPI first (other services may depend on it)
        fastapi_proc = self.start_fastapi()
        if fastapi_proc:
            time.sleep(3)  # Give FastAPI time to start
        
        # Start Streamlit
        streamlit_proc = self.start_streamlit()
        if streamlit_proc:
            time.sleep(2)  # Give Streamlit time to start
        
        # Start React frontend (optional)
        react_proc = self.start_react_frontend()
        
        # Display service status
        logger.info("\n" + "=" * 60)
        logger.info("🎯 SERVICE STATUS")
        logger.info("=" * 60)
        
        if fastapi_proc:
            logger.info("🔌 FastAPI Backend:    http://localhost:8000")
            logger.info("📚 API Documentation:  http://localhost:8000/docs")
        
        if streamlit_proc:
            logger.info("📊 Streamlit UI:       http://localhost:8501")
        
        if react_proc:
            logger.info("📱 React Frontend:     http://localhost:5173")
        
        logger.info("=" * 60)
        logger.info("✅ All services started successfully!")
        logger.info("💡 Press Ctrl+C to stop all services")
        logger.info("=" * 60)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Wait for shutdown signal
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 Keyboard interrupt received")
        
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all services gracefully"""
        logger.info("\n🛑 Shutting down services...")
        
        self.shutdown_event.set()
        
        for i, proc in enumerate(self.processes):
            try:
                logger.info(f"🔄 Stopping service {i+1}/{len(self.processes)}...")
                proc.terminate()
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=10)
                    logger.info(f"✅ Service {i+1} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Service {i+1} didn't stop gracefully, forcing...")
                    proc.kill()
                    proc.wait()
                    logger.info(f"🔨 Service {i+1} force stopped")
                    
            except Exception as e:
                logger.error(f"❌ Error stopping service {i+1}: {e}")
        
        logger.info("✅ All services stopped")
        logger.info("👋 Die Waarheid shutdown complete")

def main():
    """Main entry point"""
    try:
        manager = ServiceManager()
        manager.start_all()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
