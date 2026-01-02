"""
Deployment and DevOps utilities for Die Waarheid
Environment management, configuration validation, and deployment helpers
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validate deployment environment"""

    def __init__(self):
        """Initialize environment validator"""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_python_version(self, min_version: Tuple[int, int] = (3, 8)) -> bool:
        """
        Validate Python version

        Args:
            min_version: Minimum required version

        Returns:
            True if valid
        """
        import sys
        current = sys.version_info[:2]
        
        if current < min_version:
            self.errors.append(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"found {current[0]}.{current[1]}"
            )
            return False
        return True

    def validate_dependencies(self, requirements_file: Path) -> bool:
        """
        Validate installed dependencies

        Args:
            requirements_file: Path to requirements.txt

        Returns:
            True if all dependencies installed
        """
        try:
            with open(requirements_file) as f:
                requirements = f.readlines()

            missing = []
            for req in requirements:
                req = req.strip()
                if not req or req.startswith('#'):
                    continue

                package_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip()
                try:
                    __import__(package_name.replace('-', '_'))
                except ImportError:
                    missing.append(package_name)

            if missing:
                self.errors.append(f"Missing packages: {', '.join(missing)}")
                return False
            return True

        except Exception as e:
            self.errors.append(f"Error validating dependencies: {str(e)}")
            return False

    def validate_environment_variables(self, required_vars: List[str]) -> bool:
        """
        Validate required environment variables

        Args:
            required_vars: List of required variable names

        Returns:
            True if all variables set
        """
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            self.errors.append(f"Missing environment variables: {', '.join(missing)}")
            return False
        return True

    def validate_directories(self, required_dirs: List[Path]) -> bool:
        """
        Validate required directories exist

        Args:
            required_dirs: List of required directories

        Returns:
            True if all exist
        """
        missing = [str(d) for d in required_dirs if not d.exists()]
        
        if missing:
            self.errors.append(f"Missing directories: {', '.join(missing)}")
            return False
        return True

    def validate_permissions(self, paths: List[Path], permission: str = 'r') -> bool:
        """
        Validate file permissions

        Args:
            paths: Paths to check
            permission: Permission to check (r, w, x)

        Returns:
            True if all have permission
        """
        import os
        
        permission_map = {'r': os.R_OK, 'w': os.W_OK, 'x': os.X_OK}
        perm = permission_map.get(permission, os.R_OK)
        
        denied = [str(p) for p in paths if not os.access(p, perm)]
        
        if denied:
            self.errors.append(f"Permission denied: {', '.join(denied)}")
            return False
        return True

    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings
        }


class ConfigurationManager:
    """Manage application configuration"""

    def __init__(self, config_dir: Path):
        """
        Initialize configuration manager

        Args:
            config_dir: Configuration directory
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}

    def load_config(self, config_name: str) -> bool:
        """
        Load configuration file

        Args:
            config_name: Name of configuration

        Returns:
            True if successful
        """
        try:
            config_file = self.config_dir / f"{config_name}.json"
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                return False

            with open(config_file) as f:
                self.configs[config_name] = json.load(f)

            logger.info(f"Loaded configuration: {config_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False

    def save_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration file

        Args:
            config_name: Name of configuration
            config: Configuration dictionary

        Returns:
            True if successful
        """
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.config_dir / f"{config_name}.json"

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            self.configs[config_name] = config
            logger.info(f"Saved configuration: {config_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration"""
        return self.configs.get(config_name)

    def merge_configs(self, base: Dict, override: Dict) -> Dict:
        """
        Merge configurations

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()
        result.update(override)
        return result


class DeploymentHelper:
    """Helper utilities for deployment"""

    @staticmethod
    def create_deployment_package(
        source_dir: Path,
        output_file: Path,
        exclude_patterns: List[str] = None
    ) -> bool:
        """
        Create deployment package

        Args:
            source_dir: Source directory
            output_file: Output package file
            exclude_patterns: Patterns to exclude

        Returns:
            True if successful
        """
        try:
            import shutil
            
            if exclude_patterns is None:
                exclude_patterns = ['.git', '__pycache__', '.env', '*.pyc']

            def ignore_patterns(directory, files):
                ignored = []
                for pattern in exclude_patterns:
                    for file in files:
                        if pattern.replace('*', '') in file:
                            ignored.append(file)
                return ignored

            shutil.make_archive(
                str(output_file.with_suffix('')),
                'zip',
                source_dir,
                ignore=ignore_patterns
            )

            logger.info(f"Created deployment package: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error creating deployment package: {str(e)}")
            return False

    @staticmethod
    def run_command(command: str, cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Run shell command

        Args:
            command: Command to run
            cwd: Working directory

        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"Command succeeded: {command}")
                return True, result.stdout
            else:
                logger.error(f"Command failed: {command}")
                return False, result.stderr

        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return False, str(e)

    @staticmethod
    def install_dependencies(requirements_file: Path) -> bool:
        """
        Install Python dependencies

        Args:
            requirements_file: Path to requirements.txt

        Returns:
            True if successful
        """
        success, output = DeploymentHelper.run_command(
            f"pip install -r {requirements_file}"
        )
        
        if success:
            logger.info("Dependencies installed successfully")
        else:
            logger.error(f"Dependency installation failed: {output}")
        
        return success


class HealthCheckRunner:
    """Run health checks before deployment"""

    def __init__(self):
        """Initialize health check runner"""
        self.checks: Dict[str, callable] = {}
        self.results: Dict[str, bool] = {}

    def register_check(self, check_name: str, check_func: callable):
        """
        Register health check

        Args:
            check_name: Name of check
            check_func: Check function
        """
        self.checks[check_name] = check_func

    def run_checks(self) -> bool:
        """
        Run all health checks

        Returns:
            True if all checks pass
        """
        all_passed = True
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                self.results[check_name] = result
                
                if result:
                    logger.info(f"Health check passed: {check_name}")
                else:
                    logger.warning(f"Health check failed: {check_name}")
                    all_passed = False

            except Exception as e:
                logger.error(f"Health check error: {check_name}: {str(e)}")
                self.results[check_name] = False
                all_passed = False

        return all_passed

    def get_results(self) -> Dict[str, bool]:
        """Get check results"""
        return self.results


class RollbackManager:
    """Manage deployment rollbacks"""

    def __init__(self, backup_dir: Path):
        """
        Initialize rollback manager

        Args:
            backup_dir: Directory for backups
        """
        self.backup_dir = Path(backup_dir)
        self.backups: Dict[str, Path] = {}

    def create_backup(self, source_dir: Path, backup_name: str) -> bool:
        """
        Create backup

        Args:
            source_dir: Directory to backup
            backup_name: Name of backup

        Returns:
            True if successful
        """
        try:
            import shutil
            
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / backup_name
            
            shutil.copytree(source_dir, backup_path, dirs_exist_ok=True)
            self.backups[backup_name] = backup_path
            
            logger.info(f"Created backup: {backup_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False

    def restore_backup(self, backup_name: str, target_dir: Path) -> bool:
        """
        Restore backup

        Args:
            backup_name: Name of backup
            target_dir: Target directory

        Returns:
            True if successful
        """
        try:
            import shutil
            
            if backup_name not in self.backups:
                logger.error(f"Backup not found: {backup_name}")
                return False

            backup_path = self.backups[backup_name]
            
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.copytree(backup_path, target_dir)
            logger.info(f"Restored backup: {backup_name}")
            return True

        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
            return False

    def list_backups(self) -> List[str]:
        """List available backups"""
        return list(self.backups.keys())


class DeploymentPlan:
    """Plan and execute deployment"""

    def __init__(self, name: str):
        """
        Initialize deployment plan

        Args:
            name: Plan name
        """
        self.name = name
        self.steps: List[Dict[str, Any]] = []
        self.executed_steps: List[str] = []

    def add_step(self, step_name: str, step_func: callable, critical: bool = False):
        """
        Add deployment step

        Args:
            step_name: Name of step
            step_func: Step function
            critical: If True, failure stops deployment
        """
        self.steps.append({
            'name': step_name,
            'func': step_func,
            'critical': critical
        })

    def execute(self) -> Tuple[bool, List[str]]:
        """
        Execute deployment plan

        Returns:
            Tuple of (success, executed_steps)
        """
        for step in self.steps:
            try:
                logger.info(f"Executing step: {step['name']}")
                result = step['func']()
                
                if result:
                    self.executed_steps.append(step['name'])
                    logger.info(f"Step completed: {step['name']}")
                else:
                    logger.error(f"Step failed: {step['name']}")
                    if step['critical']:
                        return False, self.executed_steps

            except Exception as e:
                logger.error(f"Step error: {step['name']}: {str(e)}")
                if step['critical']:
                    return False, self.executed_steps

        return True, self.executed_steps


if __name__ == "__main__":
    validator = EnvironmentValidator()
    validator.validate_python_version()
    report = validator.get_validation_report()
    print(f"Validation Report: {report}")
