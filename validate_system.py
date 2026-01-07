"""
Final System Validation for Die Waarheid
Comprehensive health check and validation script
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Add die_waarheid to path
sys.path.insert(0, str(Path(__file__).parent / "die_waarheid"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemValidator:
    """Comprehensive system validator"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'validations': {},
            'overall_status': 'unknown',
            'errors': [],
            'warnings': []
        }
    
    def validate_all(self):
        """Run all validations"""
        logger.info("Starting comprehensive system validation...")
        
        validations = [
            self.validate_environment,
            self.validate_dependencies,
            self.validate_configuration,
            self.validate_directories,
            self.validate_api_connectivity,
            self.validate_core_components,
            self.validate_audio_processing,
            self.validate_ai_analysis,
            self.validate_database,
            self.validate_performance
        ]
        
        for validation in validations:
            try:
                validation()
            except Exception as e:
                logger.error(f"Validation {validation.__name__} failed: {str(e)}")
                self.results['errors'].append(f"{validation.__name__}: {str(e)}")
        
        self._calculate_overall_status()
        self._print_results()
        
        return self.results['overall_status'] == 'healthy'
    
    def validate_environment(self):
        """Validate Python environment"""
        logger.info("Validating environment...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        # Check Python version
        py_version = sys.version_info
        if py_version.major == 3 and py_version.minor >= 10:
            result['details']['python_version'] = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
        else:
            result['status'] = 'warning'
            result['details']['python_version'] = f"{py_version.major}.{py_version.minor}.{py_version.micro} (recommended: 3.10+)"
            self.results['warnings'].append("Python version < 3.10 detected")
        
        # Check platform
        result['details']['platform'] = sys.platform
        
        self.results['validations']['environment'] = result
        logger.info(f"Environment validation: {result['status']}")
    
    def validate_dependencies(self):
        """Validate required dependencies"""
        logger.info("Validating dependencies...")
        
        result = {
            'status': 'healthy',
            'details': {},
            'missing': []
        }
        
        required_packages = [
            'streamlit',
            'librosa',
            'torch',
            'transformers',
            'pandas',
            'numpy',
            'plotly',
            'sqlalchemy',
            'google.generativeai'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                result['details'][package] = 'installed'
            except ImportError:
                result['missing'].append(package)
                result['status'] = 'error'
        
        if result['missing']:
            self.results['errors'].append(f"Missing packages: {', '.join(result['missing'])}")
        
        self.results['validations']['dependencies'] = result
        logger.info(f"Dependencies validation: {result['status']}")
    
    def validate_configuration(self):
        """Validate configuration files"""
        logger.info("Validating configuration...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        # Check .env file
        env_file = Path("die_waarheid/.env")
        if env_file.exists():
            result['details']['env_file'] = 'exists'
            
            # Check required variables
            with open(env_file) as f:
                env_content = f.read()
                
            if 'GEMINI_API_KEY' in env_content:
                result['details']['gemini_key'] = 'configured'
            else:
                result['details']['gemini_key'] = 'missing'
                result['status'] = 'warning'
                self.results['warnings'].append("GEMINI_API_KEY not configured")
        else:
            result['status'] = 'error'
            result['details']['env_file'] = 'missing'
            self.results['errors'].append(".env file not found")
        
        self.results['validations']['configuration'] = result
        logger.info(f"Configuration validation: {result['status']}")
    
    def validate_directories(self):
        """Validate required directories"""
        logger.info("Validating directories...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        required_dirs = [
            'die_waarheid/data/audio',
            'die_waarheid/data/text',
            'die_waarheid/data/temp',
            'die_waarheid/data/output',
            'die_waarheid/data/logs',
            'die_waarheid/data/reports'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                result['details'][dir_path] = 'exists'
            else:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    result['details'][dir_path] = 'created'
                except Exception as e:
                    result['status'] = 'error'
                    result['details'][dir_path] = f'error: {str(e)}'
                    self.results['errors'].append(f"Cannot create directory {dir_path}")
        
        self.results['validations']['directories'] = result
        logger.info(f"Directories validation: {result['status']}")
    
    def validate_api_connectivity(self):
        """Validate API connectivity"""
        logger.info("Validating API connectivity...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        try:
            from src.ai_analyzer import AIAnalyzer
            analyzer = AIAnalyzer()
            
            if analyzer.configured:
                result['details']['gemini'] = 'connected'
            else:
                result['status'] = 'warning'
                result['details']['gemini'] = 'not configured'
                self.results['warnings'].append("Gemini API not configured")
        except Exception as e:
            result['status'] = 'error'
            result['details']['gemini'] = f'error: {str(e)}'
            self.results['errors'].append(f"API connectivity error: {str(e)}")
        
        self.results['validations']['api_connectivity'] = result
        logger.info(f"API connectivity validation: {result['status']}")
    
    def validate_core_components(self):
        """Validate core components"""
        logger.info("Validating core components...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        components = {
            'pipeline_processor': 'src.pipeline_processor.PipelineProcessor',
            'whisper_transcriber': 'src.whisper_transcriber.WhisperTranscriber',
            'forensics_engine': 'src.forensics.ForensicsEngine',
            'ai_analyzer': 'src.ai_analyzer.AIAnalyzer',
            'speaker_identification': 'src.speaker_identification.SpeakerIdentificationSystem'
        }
        
        for name, import_path in components.items():
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                result['details'][name] = 'available'
            except Exception as e:
                result['status'] = 'error'
                result['details'][name] = f'error: {str(e)}'
                self.results['errors'].append(f"Component {name} error: {str(e)}")
        
        self.results['validations']['core_components'] = result
        logger.info(f"Core components validation: {result['status']}")
    
    def validate_audio_processing(self):
        """Validate audio processing capabilities"""
        logger.info("Validating audio processing...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        try:
            import librosa
            import soundfile as sf
            
            # Test audio loading
            y, sr = librosa.load(duration=1)
            result['details']['librosa'] = 'working'
            result['details']['sample_rate'] = sr
            
        except Exception as e:
            result['status'] = 'error'
            result['details']['audio_processing'] = f'error: {str(e)}'
            self.results['errors'].append(f"Audio processing error: {str(e)}")
        
        self.results['validations']['audio_processing'] = result
        logger.info(f"Audio processing validation: {result['status']}")
    
    def validate_ai_analysis(self):
        """Validate AI analysis functionality"""
        logger.info("Validating AI analysis...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        try:
            from src.ai_analyzer import AIAnalyzer
            analyzer = AIAnalyzer()
            
            # Test fallback analysis
            test_text = "This is a test message"
            fallback_result = analyzer.analyze_message_fallback(test_text)
            
            if fallback_result['success']:
                result['details']['fallback_analysis'] = 'working'
                result['details']['emotion_detection'] = fallback_result.get('emotion', 'unknown')
            else:
                result['status'] = 'warning'
                result['details']['fallback_analysis'] = 'not working'
                self.results['warnings'].append("Fallback analysis not working")
            
        except Exception as e:
            result['status'] = 'error'
            result['details']['ai_analysis'] = f'error: {str(e)}'
            self.results['errors'].append(f"AI analysis error: {str(e)}")
        
        self.results['validations']['ai_analysis'] = result
        logger.info(f"AI analysis validation: {result['status']}")
    
    def validate_database(self):
        """Validate database connectivity"""
        logger.info("Validating database...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        try:
            from src.speaker_identification import SpeakerIdentificationSystem
            speaker_system = SpeakerIdentificationSystem("validation_test")
            
            # Test database operations
            participants = speaker_system.get_all_participants()
            result['details']['speaker_db'] = 'connected'
            result['details']['participants_count'] = len(participants)
            
            # Cleanup
            speaker_system.close()
            
        except Exception as e:
            result['status'] = 'warning'
            result['details']['database'] = f'warning: {str(e)}'
            self.results['warnings'].append(f"Database warning: {str(e)}")
        
        self.results['validations']['database'] = result
        logger.info(f"Database validation: {result['status']}")
    
    def validate_performance(self):
        """Validate performance metrics"""
        logger.info("Validating performance...")
        
        result = {
            'status': 'healthy',
            'details': {}
        }
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            result['details']['cpu_usage'] = f"{cpu_percent}%"
            
            # Memory usage
            memory = psutil.virtual_memory()
            result['details']['memory_usage'] = f"{memory.percent}%"
            
            # Disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            result['details']['disk_free'] = f"{free_gb:.1f}GB"
            
            # Check thresholds
            if cpu_percent > 80:
                result['status'] = 'warning'
                self.results['warnings'].append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 80:
                result['status'] = 'warning'
                self.results['warnings'].append(f"High memory usage: {memory.percent}%")
            
            if free_gb < 5:
                result['status'] = 'warning'
                self.results['warnings'].append(f"Low disk space: {free_gb:.1f}GB")
            
        except Exception as e:
            result['status'] = 'error'
            result['details']['performance'] = f'error: {str(e)}'
            self.results['errors'].append(f"Performance check error: {str(e)}")
        
        self.results['validations']['performance'] = result
        logger.info(f"Performance validation: {result['status']}")
    
    def _calculate_overall_status(self):
        """Calculate overall system status"""
        statuses = [v.get('status', 'unknown') for v in self.results['validations'].values()]
        
        if 'error' in statuses:
            self.results['overall_status'] = 'unhealthy'
        elif 'warning' in statuses:
            self.results['overall_status'] = 'warning'
        else:
            self.results['overall_status'] = 'healthy'
    
    def _print_results(self):
        """Print validation results"""
        print("\n" + "="*60)
        print("DIE WAARHEID SYSTEM VALIDATION REPORT")
        print("="*60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print("-"*60)
        
        for validation_name, result in self.results['validations'].items():
            status_icon = {
                'healthy': '✅',
                'warning': '⚠️',
                'error': '❌',
                'unknown': '❓'
            }.get(result['status'], '❓')
            
            print(f"\n{status_icon} {validation_name.replace('_', ' ').title()}: {result['status'].upper()}")
            
            if result.get('details'):
                for key, value in result['details'].items():
                    print(f"   • {key}: {value}")
        
        if self.results['warnings']:
            print(f"\n⚠️ WARNINGS ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
        
        if self.results['errors']:
            print(f"\n❌ ERRORS ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        print("\n" + "="*60)
        
        # Save report
        report_file = Path(f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nReport saved to: {report_file}")


def main():
    """Main validation function"""
    validator = SystemValidator()
    success = validator.validate_all()
    
    if success:
        logger.info("✅ System validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ System validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
