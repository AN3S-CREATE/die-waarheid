"""
Advanced features and extensions for Die Waarheid
Plugin system, custom analyzers, and extensibility framework
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Type
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisPlugin(ABC):
    """Base class for analysis plugins"""

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize plugin

        Args:
            name: Plugin name
            version: Plugin version
        """
        self.name = name
        self.version = version
        self.enabled = True
        self.metadata = {}

    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Perform analysis

        Args:
            data: Data to analyze

        Returns:
            Analysis results
        """
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate input data

        Args:
            data: Data to validate

        Returns:
            True if valid
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'metadata': self.metadata
        }


class CustomAnalyzer(AnalysisPlugin):
    """Custom analyzer plugin template"""

    def __init__(self, name: str, analyzer_func: Callable):
        """
        Initialize custom analyzer

        Args:
            name: Analyzer name
            analyzer_func: Analysis function
        """
        super().__init__(name)
        self.analyzer_func = analyzer_func

    def analyze(self, data: Any) -> Dict[str, Any]:
        """Execute custom analysis"""
        try:
            result = self.analyzer_func(data)
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Custom analyzer {self.name} failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def validate(self, data: Any) -> bool:
        """Validate input"""
        return data is not None


class PluginManager:
    """Manage analysis plugins"""

    def __init__(self):
        """Initialize plugin manager"""
        self.plugins: Dict[str, AnalysisPlugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}

    def register_plugin(self, plugin: AnalysisPlugin) -> bool:
        """
        Register a plugin

        Args:
            plugin: Plugin to register

        Returns:
            True if successful
        """
        try:
            self.plugins[plugin.name] = plugin
            logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
            return True
        except Exception as e:
            logger.error(f"Error registering plugin: {str(e)}")
            return False

    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin

        Args:
            plugin_name: Name of plugin to unregister

        Returns:
            True if successful
        """
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
        return False

    def get_plugin(self, plugin_name: str) -> Optional[AnalysisPlugin]:
        """
        Get plugin by name

        Args:
            plugin_name: Name of plugin

        Returns:
            Plugin or None
        """
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins

        Returns:
            List of plugin information
        """
        return [plugin.get_info() for plugin in self.plugins.values()]

    def execute_plugin(self, plugin_name: str, data: Any) -> Dict[str, Any]:
        """
        Execute plugin analysis

        Args:
            plugin_name: Name of plugin
            data: Data to analyze

        Returns:
            Analysis results
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return {'success': False, 'error': f'Plugin {plugin_name} not found'}

        if not plugin.enabled:
            return {'success': False, 'error': f'Plugin {plugin_name} is disabled'}

        if not plugin.validate(data):
            return {'success': False, 'error': f'Invalid data for plugin {plugin_name}'}

        return plugin.analyze(data)

    def register_hook(self, hook_name: str, callback: Callable):
        """
        Register a hook callback

        Args:
            hook_name: Name of hook
            callback: Callback function
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """
        Trigger a hook

        Args:
            hook_name: Name of hook
            *args: Hook arguments
            **kwargs: Hook keyword arguments
        """
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Hook {hook_name} callback failed: {str(e)}")


class ReportTemplate:
    """Custom report template"""

    def __init__(self, name: str, template_format: str = "markdown"):
        """
        Initialize report template

        Args:
            name: Template name
            template_format: Template format (markdown, html, json)
        """
        self.name = name
        self.template_format = template_format
        self.sections: List[Dict[str, Any]] = []

    def add_section(self, title: str, content: str, section_type: str = "text"):
        """
        Add section to template

        Args:
            title: Section title
            content: Section content
            section_type: Type of section
        """
        self.sections.append({
            'title': title,
            'content': content,
            'type': section_type
        })

    def render(self, data: Dict[str, Any]) -> str:
        """
        Render template with data

        Args:
            data: Data to render

        Returns:
            Rendered template
        """
        if self.template_format == "markdown":
            return self._render_markdown(data)
        elif self.template_format == "html":
            return self._render_html(data)
        else:
            return self._render_json(data)

    def _render_markdown(self, data: Dict[str, Any]) -> str:
        """Render as markdown"""
        output = f"# {self.name}\n\n"
        for section in self.sections:
            output += f"## {section['title']}\n\n"
            output += f"{section['content']}\n\n"
        return output

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render as HTML"""
        output = f"<h1>{self.name}</h1>\n"
        for section in self.sections:
            output += f"<h2>{section['title']}</h2>\n"
            output += f"<p>{section['content']}</p>\n"
        return output

    def _render_json(self, data: Dict[str, Any]) -> str:
        """Render as JSON"""
        import json
        return json.dumps({
            'name': self.name,
            'sections': self.sections,
            'data': data
        }, indent=2)


class DataTransformer:
    """Transform data between formats"""

    @staticmethod
    def transform_forensics_to_summary(forensics: Dict[str, Any]) -> Dict[str, str]:
        """
        Transform forensics data to summary

        Args:
            forensics: Forensics data

        Returns:
            Summary dictionary
        """
        return {
            'filename': forensics.get('filename', 'Unknown'),
            'duration': f"{forensics.get('duration', 0):.1f}s",
            'stress_level': f"{forensics.get('stress_level', 0):.1f}/100",
            'pitch_volatility': f"{forensics.get('pitch_volatility', 0):.1f}",
            'silence_ratio': f"{forensics.get('silence_ratio', 0):.1%}",
            'status': 'High Stress' if forensics.get('stress_level', 0) > 50 else 'Normal'
        }

    @staticmethod
    def transform_profile_to_summary(profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Transform psychological profile to summary

        Args:
            profile: Profile data

        Returns:
            Summary dictionary
        """
        return {
            'traits': ', '.join(profile.get('personality_traits', [])),
            'emotional_regulation': profile.get('emotional_regulation', 'Unknown'),
            'risk_level': profile.get('risk_assessment', 'Unknown'),
            'communication': ', '.join(profile.get('communication_patterns', []))
        }

    @staticmethod
    def flatten_nested_dict(data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict:
        """
        Flatten nested dictionary

        Args:
            data: Nested dictionary
            parent_key: Parent key prefix
            sep: Separator for keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DataTransformer.flatten_nested_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ExportManager:
    """Manage data exports"""

    def __init__(self):
        """Initialize export manager"""
        self.exporters: Dict[str, Callable] = {}

    def register_exporter(self, format_name: str, exporter_func: Callable):
        """
        Register export format

        Args:
            format_name: Format name
            exporter_func: Export function
        """
        self.exporters[format_name] = exporter_func
        logger.info(f"Registered exporter: {format_name}")

    def export(self, data: Any, format_name: str, output_path: Path) -> bool:
        """
        Export data

        Args:
            data: Data to export
            format_name: Export format
            output_path: Output file path

        Returns:
            True if successful
        """
        if format_name not in self.exporters:
            logger.error(f"Exporter {format_name} not found")
            return False

        try:
            exporter = self.exporters[format_name]
            content = exporter(data)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False

    def list_exporters(self) -> List[str]:
        """List available exporters"""
        return list(self.exporters.keys())


class ExtensionLoader:
    """Load and manage extensions"""

    def __init__(self, extension_dir: Path):
        """
        Initialize extension loader

        Args:
            extension_dir: Directory containing extensions
        """
        self.extension_dir = Path(extension_dir)
        self.loaded_extensions: Dict[str, Any] = {}

    def load_extension(self, extension_name: str) -> bool:
        """
        Load extension from file

        Args:
            extension_name: Name of extension

        Returns:
            True if successful
        """
        try:
            extension_file = self.extension_dir / f"{extension_name}.py"
            if not extension_file.exists():
                logger.error(f"Extension file not found: {extension_file}")
                return False

            import importlib.util
            spec = importlib.util.spec_from_file_location(extension_name, extension_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.loaded_extensions[extension_name] = module
            logger.info(f"Loaded extension: {extension_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading extension {extension_name}: {str(e)}")
            return False

    def get_extension(self, extension_name: str) -> Optional[Any]:
        """
        Get loaded extension

        Args:
            extension_name: Name of extension

        Returns:
            Extension module or None
        """
        return self.loaded_extensions.get(extension_name)

    def list_extensions(self) -> List[str]:
        """List loaded extensions"""
        return list(self.loaded_extensions.keys())


if __name__ == "__main__":
    manager = PluginManager()
    
    def custom_analysis(data):
        return {"custom_result": len(str(data))}
    
    plugin = CustomAnalyzer("test_analyzer", custom_analysis)
    manager.register_plugin(plugin)
    
    result = manager.execute_plugin("test_analyzer", "test data")
    print(f"Result: {result}")
    print(f"Plugins: {manager.list_plugins()}")
