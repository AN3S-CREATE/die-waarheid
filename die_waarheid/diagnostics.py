"""
Full system diagnostics for Die Waarheid
Tests all components and reports status
"""

import sys
import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SystemDiagnostics:
    def __init__(self):
        self.results = {
            "python_version": sys.version,
            "environment": {},
            "imports": {},
            "files": {},
            "api_keys": {},
            "gpu": {},
            "memory": {},
            "recommendations": []
        }
    
    def check_environment(self):
        """Check Python environment and packages."""
        logger.info("🔍 Checking environment...")
        
        # Check Python version
        self.results["environment"]["python"] = {
            "version": sys.version,
            "version_info": str(sys.version_info),
            "recommended": "3.11.x or 3.12.x",
            "status": "OK" if sys.version_info >= (3, 11) else "WARNING"
        }
        
        # Check key packages
        packages = {
            "streamlit": "1.52.2",
            "librosa": "0.10.1",
            "numpy": "1.24.3",
            "pandas": "2.0.3",
            "google-generativeai": "0.3.2",
            "openai": "1.3.5",
            "whisper": None,
            "torch": None,
            "torchaudio": None,
            "transformers": None,
            "ffmpeg-python": "0.2.0"
        }
        
        for package, min_version in packages.items():
            try:
                if package == "whisper":
                    import whisper
                    version = whisper.__version__
                elif package == "openai":
                    import openai
                    version = openai.__version__
                else:
                    module = __import__(package.replace("-", "_"))
                    version = getattr(module, "__version__", "unknown")
                
                self.results["environment"][package] = {
                    "installed": True,
                    "version": version,
                    "status": "OK"
                }
            except ImportError as e:
                self.results["environment"][package] = {
                    "installed": False,
                    "error": str(e),
                    "status": "MISSING"
                }
                self.results["recommendations"].append(f"Install {package}: pip install {package}")
    
    def check_imports(self):
        """Check if all modules can be imported."""
        logger.info("📦 Checking imports...")
        
        modules = [
            "src.whisper_transcriber",
            "src.afrikaans_processor",
            "src.diarization",
            "src.multilingual_support",
            "src.audio_separation",
            "src.background_sound_analyzer",
            "src.text_forensics",
            "src.narrative_reconstruction",
            "src.contradiction_timeline",
            "src.risk_escalation_matrix",
            "src.investigative_checklist",
            "src.expert_panel",
            "real_analysis_engine",
            "verification_harness",
            "ingestion_engine"
        ]
        
        for module in modules:
            try:
                __import__(module)
                self.results["imports"][module] = {"status": "OK"}
            except Exception as e:
                self.results["imports"][module] = {
                    "status": "FAILED",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
    
    def check_files(self):
        """Check if required files exist."""
        logger.info("📁 Checking files...")
        
        required_files = [
            "dashboard_complete.py",
            "dashboard_real.py",
            "session_data.json",
            "real_analysis_engine.py",
            "verification_harness.py",
            "ingestion_engine.py",
            "automated_tests.py",
            "FORENSIC_VALIDATION.md",
            "processing_time_estimator.py"
        ]
        
        for file in required_files:
            path = Path(file)
            self.results["files"][file] = {
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "status": "OK" if path.exists() else "MISSING"
            }
    
    def check_api_keys(self):
        """Check API key configuration."""
        logger.info("🔑 Checking API keys...")
        
        # Check environment variables
        api_vars = {
            "GOOGLE_API_KEY": "Google Generative AI (Gemini)",
            "OPENAI_API_KEY": "OpenAI GPT",
            "HUGGINGFACE_API_KEY": "Hugging Face"
        }
        
        for var, name in api_vars.items():
            value = os.getenv(var)
            self.results["api_keys"][var] = {
                "name": name,
                "configured": value is not None and len(value) > 0,
                "status": "OK" if value else "NOT SET"
            }
            
            if not value:
                self.results["recommendations"].append(f"Set {var} environment variable for {name}")
    
    def check_gpu(self):
        """Check GPU availability."""
        logger.info("🎮 Checking GPU...")
        
        try:
            import torch
            self.results["gpu"]["pytorch"] = {
                "available": True,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except ImportError:
            self.results["gpu"]["pytorch"] = {"available": False}
            self.results["recommendations"].append("Install PyTorch for GPU acceleration: pip install torch")
    
    def check_memory(self):
        """Check memory usage."""
        logger.info("💾 Checking memory...")
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.results["memory"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent,
                "status": "OK" if memory.available > 2 * (1024**3) else "LOW"
            }
            
            if memory.available < 2 * (1024**3):
                self.results["recommendations"].append("Low memory - consider closing other applications")
        except ImportError:
            self.results["memory"] = {"error": "psutil not installed"}
    
    def test_whisper(self):
        """Test Whisper transcription."""
        logger.info("🎤 Testing Whisper...")
        
        try:
            import whisper
            model = whisper.load_model("base")
            self.results["whisper"] = {
                "status": "OK",
                "model_loaded": True,
                "model_size": "base"
            }
        except Exception as e:
            self.results["whisper"] = {
                "status": "FAILED",
                "error": str(e)
            }
            self.results["recommendations"].append("Whisper failed to load - check installation")
    
    def test_google_ai(self):
        """Test Google AI connection."""
        logger.info("🤖 Testing Google AI...")
        
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                self.results["google_ai"] = {
                    "status": "NO_KEY",
                    "error": "GOOGLE_API_KEY not set"
                }
                return
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Simple test
            response = model.generate_content("Test")
            
            self.results["google_ai"] = {
                "status": "OK",
                "model": "gemini-pro",
                "response_length": len(response.text) if hasattr(response, 'text') else 0
            }
        except Exception as e:
            self.results["google_ai"] = {
                "status": "FAILED",
                "error": str(e)
            }
            self.results["recommendations"].append("Google AI connection failed - check API key and internet")
    
    def generate_report(self) -> str:
        """Generate diagnostic report."""
        report = []
        report.append("=" * 70)
        report.append("🔍 DIE WAARHEID SYSTEM DIAGNOSTICS")
        report.append("=" * 70)
        report.append("")
        
        # Environment
        report.append("🐍 Python Environment")
        report.append("-" * 40)
        py_info = self.results["environment"].get("python", {})
        report.append(f"Version: {py_info.get('version', 'Unknown')}")
        report.append(f"Status: {py_info.get('status', 'Unknown')}")
        report.append(f"Recommended: {py_info.get('recommended', 'Unknown')}")
        report.append("")
        
        # Package status
        report.append("📦 Package Status")
        report.append("-" * 40)
        for pkg, info in self.results["environment"].items():
            if pkg == "python":
                continue
            status = info.get("status", "UNKNOWN")
            version = info.get("version", "N/A")
            report.append(f"{pkg:20} {status:10} {version}")
        report.append("")
        
        # Import status
        report.append("📥 Import Status")
        report.append("-" * 40)
        for module, info in self.results["imports"].items():
            status = info.get("status", "UNKNOWN")
            icon = "✅" if status == "OK" else "❌"
            report.append(f"{icon} {module}")
            if status == "FAILED":
                report.append(f"   Error: {info.get('error', 'Unknown')}")
        report.append("")
        
        # API status
        report.append("🔑 API Configuration")
        report.append("-" * 40)
        for api, info in self.results["api_keys"].items():
            status = info.get("status", "UNKNOWN")
            icon = "✅" if status == "OK" else "❌"
            report.append(f"{icon} {info.get('name', api)}: {status}")
        report.append("")
        
        # GPU status
        gpu_info = self.results.get("gpu", {})
        if gpu_info.get("pytorch", {}).get("available"):
            report.append("🎮 GPU Status")
            report.append("-" * 40)
            report.append(f"CUDA Available: {gpu_info['pytorch']['cuda_available']}")
            if gpu_info['pytorch']['cuda_available']:
                report.append(f"GPU Count: {gpu_info['pytorch']['cuda_device_count']}")
                report.append(f"GPU Name: {gpu_info['pytorch']['device_name']}")
        report.append("")
        
        # Memory status
        mem_info = self.results.get("memory", {})
        if "total_gb" in mem_info:
            report.append("💾 Memory Status")
            report.append("-" * 40)
            report.append(f"Total: {mem_info['total_gb']:.1f} GB")
            report.append(f"Available: {mem_info['available_gb']:.1f} GB")
            report.append(f"Used: {mem_info['percent_used']:.1f}%")
            report.append(f"Status: {mem_info['status']}")
        report.append("")
        
        # Test results
        if "whisper" in self.results:
            report.append("🎤 Whisper Test")
            report.append("-" * 40)
            whisper = self.results["whisper"]
            report.append(f"Status: {whisper.get('status', 'Unknown')}")
            if whisper.get("status") == "OK":
                report.append(f"Model: {whisper.get('model_size', 'Unknown')}")
            else:
                report.append(f"Error: {whisper.get('error', 'Unknown')}")
        report.append("")
        
        if "google_ai" in self.results:
            report.append("🤖 Google AI Test")
            report.append("-" * 40)
            ai = self.results["google_ai"]
            report.append(f"Status: {ai.get('status', 'Unknown')}")
            if ai.get("status") == "OK":
                report.append(f"Model: {ai.get('model', 'Unknown')}")
            else:
                report.append(f"Error: {ai.get('error', 'Unknown')}")
        report.append("")
        
        # Recommendations
        if self.results["recommendations"]:
            report.append("💡 Recommendations")
            report.append("-" * 40)
            for i, rec in enumerate(self.results["recommendations"], 1):
                report.append(f"{i}. {rec}")
        report.append("")
        
        # Summary
        report.append("📊 Summary")
        report.append("-" * 40)
        total_imports = len(self.results["imports"])
        ok_imports = sum(1 for i in self.results["imports"].values() if i.get("status") == "OK")
        report.append(f"Imports: {ok_imports}/{total_imports} working")
        
        total_apis = len(self.results["api_keys"])
        ok_apis = sum(1 for a in self.results["api_keys"].values() if a.get("status") == "OK")
        report.append(f"APIs: {ok_apis}/{total_apis} configured")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        logger.info("🚀 Running full diagnostics...")
        
        self.check_environment()
        self.check_imports()
        self.check_files()
        self.check_api_keys()
        self.check_gpu()
        self.check_memory()
        self.test_whisper()
        self.test_google_ai()
        
        return self.results


def main():
    """Run diagnostics and print report."""
    diagnostics = SystemDiagnostics()
    diagnostics.run_all_tests()
    
    # Print report
    print(diagnostics.generate_report())
    
    # Save detailed results
    output_file = Path("diagnostics_report.json")
    with open(output_file, 'w') as f:
        json.dump(diagnostics.results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
