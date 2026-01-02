"""
Report Generator for Die Waarheid
Generates comprehensive forensic analysis reports in multiple formats
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

import pandas as pd

from config import (
    REPORT_TEMPLATE,
    REPORTS_DIR,
    EXPORTS_DIR
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive forensic analysis reports
    Supports markdown, HTML, JSON, and CSV export formats
    """

    def __init__(self):
        self.report_data = {}
        self.case_id = None
        self.generated_at = None

    def set_case_info(
        self,
        case_id: str,
        date_range: Tuple[datetime, datetime],
        participants: List[str],
        total_messages: int,
        total_audio: int
    ) -> None:
        """
        Set case information

        Args:
            case_id: Case identifier
            date_range: Tuple of (start_date, end_date)
            participants: List of participant names
            total_messages: Total messages analyzed
            total_audio: Total audio files analyzed
        """
        self.case_id = case_id
        self.generated_at = datetime.now()

        self.report_data['case_id'] = case_id
        self.report_data['date_range'] = f"{date_range[0].date()} to {date_range[1].date()}"
        self.report_data['participants'] = ', '.join(participants)
        self.report_data['total_messages'] = total_messages
        self.report_data['total_audio'] = total_audio
        self.report_data['timestamp'] = self.generated_at.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Set case info for {case_id}")

    def add_executive_summary(
        self,
        trust_score: float,
        critical_findings: List[str],
        key_metrics: Dict
    ) -> None:
        """
        Add executive summary section

        Args:
            trust_score: Trust score (0-100)
            critical_findings: List of critical findings
            key_metrics: Dictionary with key metrics
        """
        trust_interpretation = self._interpret_trust_score(trust_score)

        self.report_data['trust_score'] = trust_score
        self.report_data['trust_interpretation'] = trust_interpretation
        self.report_data['critical_findings'] = '\n'.join([f"- {f}" for f in critical_findings])
        self.report_data['metrics_summary'] = self._format_metrics(key_metrics)

        logger.info(f"Added executive summary with trust score: {trust_score}")

    def add_psychological_profile(self, profile_data: Dict) -> None:
        """
        Add psychological profile section

        Args:
            profile_data: Dictionary with profile information
        """
        profile_text = "## Psychological Profile\n\n"

        if 'personality_traits' in profile_data:
            profile_text += "**Personality Traits:**\n"
            for trait in profile_data['personality_traits']:
                profile_text += f"- {trait}\n"
            profile_text += "\n"

        if 'communication_patterns' in profile_data:
            profile_text += "**Communication Patterns:**\n"
            for pattern in profile_data['communication_patterns']:
                profile_text += f"- {pattern}\n"
            profile_text += "\n"

        if 'stress_indicators' in profile_data:
            profile_text += "**Stress Indicators:**\n"
            for indicator in profile_data['stress_indicators']:
                profile_text += f"- {indicator}\n"
            profile_text += "\n"

        self.report_data['psychological_profile'] = profile_text

        logger.info("Added psychological profile section")

    def add_contradictions(self, contradictions: List[Dict]) -> None:
        """
        Add detected contradictions section

        Args:
            contradictions: List of contradiction dictionaries
        """
        if not contradictions:
            self.report_data['contradictions'] = "No contradictions detected."
            return

        contradiction_text = "## Detected Contradictions\n\n"

        for idx, contradiction in enumerate(contradictions, 1):
            contradiction_text += f"### Contradiction {idx}\n\n"
            contradiction_text += f"**Statement 1:** {contradiction.get('statement1', 'N/A')}\n\n"
            contradiction_text += f"**Statement 2:** {contradiction.get('statement2', 'N/A')}\n\n"
            contradiction_text += f"**Explanation:** {contradiction.get('explanation', 'N/A')}\n\n"

        self.report_data['contradictions'] = contradiction_text

        logger.info(f"Added {len(contradictions)} contradictions")

    def add_biosignal_analysis(self, biosignal_data: Dict) -> None:
        """
        Add bio-signal analysis section

        Args:
            biosignal_data: Dictionary with bio-signal metrics
        """
        biosignal_text = "## Bio-Signal Analysis\n\n"

        if 'average_stress' in biosignal_data:
            biosignal_text += f"**Average Stress Level:** {biosignal_data['average_stress']:.2f}/100\n\n"

        if 'high_stress_count' in biosignal_data:
            biosignal_text += f"**High Stress Instances:** {biosignal_data['high_stress_count']}\n\n"

        if 'avg_pitch_volatility' in biosignal_data:
            biosignal_text += f"**Average Pitch Volatility:** {biosignal_data['avg_pitch_volatility']:.2f}\n\n"

        if 'avg_silence_ratio' in biosignal_data:
            biosignal_text += f"**Average Silence Ratio:** {biosignal_data['avg_silence_ratio']:.2f}\n\n"

        self.report_data['biosignal_summary'] = biosignal_text

        logger.info("Added bio-signal analysis section")

    def add_recommendations(self, recommendations: List[str]) -> None:
        """
        Add recommendations section

        Args:
            recommendations: List of recommendations
        """
        recommendations_text = "## Recommendations\n\n"

        for recommendation in recommendations:
            recommendations_text += f"- {recommendation}\n"

        self.report_data['recommendations'] = recommendations_text

        logger.info(f"Added {len(recommendations)} recommendations")

    def _interpret_trust_score(self, score: float) -> str:
        """
        Interpret trust score

        Args:
            score: Trust score (0-100)

        Returns:
            Interpretation string
        """
        if score >= 80:
            return "High trust - Communication appears consistent and reliable."
        elif score >= 60:
            return "Moderate trust - Some inconsistencies detected but generally reliable."
        elif score >= 40:
            return "Low trust - Multiple inconsistencies and concerning patterns detected."
        else:
            return "Very low trust - Significant inconsistencies and reliability concerns."

    def _format_metrics(self, metrics: Dict) -> str:
        """
        Format metrics for display

        Args:
            metrics: Dictionary with metrics

        Returns:
            Formatted metrics string
        """
        metrics_text = "### Key Metrics\n\n"

        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            metrics_text += f"- **{formatted_key}:** {value}\n"

        return metrics_text

    def generate_markdown_report(self) -> str:
        """
        Generate markdown formatted report

        Returns:
            Markdown formatted report string
        """
        report = REPORT_TEMPLATE.format(**self.report_data)
        logger.info("Generated markdown report")
        return report

    def generate_html_report(self) -> str:
        """
        Generate HTML formatted report

        Returns:
            HTML formatted report string
        """
        markdown_report = self.generate_markdown_report()

        try:
            import markdown
            html_content = markdown.markdown(markdown_report)
        except ImportError:
            logger.warning("markdown library not available, using basic HTML conversion")
            html_content = markdown_report.replace('\n', '<br>')

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Die Waarheid - Forensic Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            border-bottom: 3px solid #FF6B6B;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metric {{
            background-color: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #4ECDC4;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #FF6B6B;
            padding: 10px;
            margin: 10px 0;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕵️ Die Waarheid - Forensic Analysis Report</h1>
            <p>Generated: {self.report_data.get('timestamp', 'N/A')}</p>
        </div>
        <div class="content">
            {html_content}
        </div>
        <div class="footer">
            <p><strong>Legal Disclaimer:</strong> This report is generated by AI-assisted analysis and should not be considered as sole evidence in legal proceedings. All findings should be verified by qualified professionals.</p>
            <p>Report ID: {self.case_id or 'N/A'}</p>
        </div>
    </div>
</body>
</html>
"""

        logger.info("Generated HTML report")
        return html

    def generate_json_report(self) -> str:
        """
        Generate JSON formatted report

        Returns:
            JSON formatted report string
        """
        report_dict = {
            'metadata': {
                'case_id': self.case_id,
                'generated_at': self.generated_at.isoformat() if self.generated_at else None,
                'report_type': 'forensic_analysis'
            },
            'case_info': {
                'case_id': self.report_data.get('case_id'),
                'date_range': self.report_data.get('date_range'),
                'participants': self.report_data.get('participants'),
                'total_messages': self.report_data.get('total_messages'),
                'total_audio': self.report_data.get('total_audio')
            },
            'analysis': {
                'trust_score': self.report_data.get('trust_score'),
                'trust_interpretation': self.report_data.get('trust_interpretation'),
                'critical_findings': self.report_data.get('critical_findings'),
                'contradictions': self.report_data.get('contradictions')
            }
        }

        logger.info("Generated JSON report")
        return json.dumps(report_dict, indent=2, default=str)

    def export_to_markdown(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Export report to markdown file

        Args:
            output_path: Path to save report (uses default if None)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if output_path is None:
                output_path = REPORTS_DIR / f"report_{self.case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = self.generate_markdown_report()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Exported markdown report to {output_path}")
            return True, f"Report exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting markdown report: {str(e)}")
            return False, f"Export error: {str(e)}"

    def export_to_html(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Export report to HTML file

        Args:
            output_path: Path to save report (uses default if None)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if output_path is None:
                output_path = REPORTS_DIR / f"report_{self.case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = self.generate_html_report()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Exported HTML report to {output_path}")
            return True, f"Report exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting HTML report: {str(e)}")
            return False, f"Export error: {str(e)}"

    def export_to_json(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Export report to JSON file

        Args:
            output_path: Path to save report (uses default if None)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if output_path is None:
                output_path = REPORTS_DIR / f"report_{self.case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = self.generate_json_report()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Exported JSON report to {output_path}")
            return True, f"Report exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting JSON report: {str(e)}")
            return False, f"Export error: {str(e)}"

    def export_all_formats(self, output_dir: Optional[Path] = None) -> Dict[str, Tuple[bool, str]]:
        """
        Export report in all formats

        Args:
            output_dir: Directory to save reports (uses default if None)

        Returns:
            Dictionary with export results for each format
        """
        if output_dir is None:
            output_dir = REPORTS_DIR

        results = {
            'markdown': self.export_to_markdown(output_dir / f"report_{self.case_id}.md"),
            'html': self.export_to_html(output_dir / f"report_{self.case_id}.html"),
            'json': self.export_to_json(output_dir / f"report_{self.case_id}.json")
        }

        logger.info(f"Exported report in all formats to {output_dir}")
        return results

    def get_report_summary(self) -> Dict:
        """
        Get report summary

        Returns:
            Dictionary with report summary
        """
        return {
            'case_id': self.case_id,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
            'trust_score': self.report_data.get('trust_score'),
            'total_messages': self.report_data.get('total_messages'),
            'total_audio': self.report_data.get('total_audio'),
            'participants': self.report_data.get('participants')
        }


if __name__ == "__main__":
    generator = ReportGenerator()
    print("Report Generator initialized")
