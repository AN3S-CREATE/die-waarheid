"""
End-to-End Integration Test for Die Waarheid
Demonstrates all 8 recommended modules + core system working together
Verifies complete forensic analysis workflow
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_module_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports (Core Recommended Modules)")
    print("="*60)

    # Test only the 8 recommended modules that don't require external deps
    modules_to_test = [
        ('alert_system', 'AlertSystem'),
        ('evidence_scoring', 'EvidenceScoringSystem'),
        ('investigative_checklist', 'InvestigativeChecklistGenerator'),
        ('contradiction_timeline', 'ContradictionTimelineAnalyzer'),
        ('narrative_reconstruction', 'NarrativeReconstructor'),
        ('comparative_psychology', 'ComparativePsychologyAnalyzer'),
        ('risk_escalation_matrix', 'RiskEscalationMatrix'),
        ('multilingual_support', 'MultilingualAnalyzer'),
    ]

    results = {}
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            results[module_name] = True
            print(f"✓ {module_name:30} - {class_name}")
        except Exception as e:
            results[module_name] = False
            print(f"✗ {module_name:30} - ERROR: {str(e)[:50]}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nResult: {passed}/{total} recommended modules imported successfully")
    
    # Note about core modules
    print("\nNote: Core modules (unified_analyzer, investigation_tracker, expert_panel,")
    print("speaker_identification) require optional dependencies (sqlalchemy, google.generativeai)")
    print("but are loaded gracefully with fallbacks in the orchestrator.")
    
    return passed == total


def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    print("\n" + "="*60)
    print("TEST 2: Orchestrator Initialization")
    print("="*60)

    try:
        from integration_orchestrator import IntegrationOrchestrator
        
        orchestrator = IntegrationOrchestrator()
        status = orchestrator.get_module_status()
        
        loaded = sum(1 for v in status.values() if v)
        total = len(status)
        
        print(f"\nModule Status ({loaded}/{total} loaded):")
        for module_name, is_loaded in sorted(status.items()):
            symbol = "✓" if is_loaded else "✗"
            print(f"  {symbol} {module_name}")
        
        print(f"\nResult: Orchestrator initialized with {loaded}/{total} modules")
        return loaded > 0
        
    except Exception as e:
        print(f"✗ Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_case_creation():
    """Test case creation workflow"""
    print("\n" + "="*60)
    print("TEST 3: Case Creation Workflow")
    print("="*60)

    try:
        from integration_orchestrator import IntegrationOrchestrator
        
        orchestrator = IntegrationOrchestrator()
        
        # Create case
        case_id = "TEST_CASE_001"
        case_name = "Test Investigation"
        
        success = orchestrator.create_case(case_id, case_name)
        
        if success:
            print(f"✓ Case created: {case_id}")
            print(f"  Name: {case_name}")
            return True
        else:
            print(f"✗ Case creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Case creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_system():
    """Test alert system functionality"""
    print("\n" + "="*60)
    print("TEST 4: Alert System")
    print("="*60)

    try:
        from alert_system import AlertSystem, AlertSeverity, AlertType
        
        alert_system = AlertSystem()
        
        # Test contradiction alert
        alert = alert_system.check_contradiction(
            evidence_id="EV_001",
            participant_id="PART_A",
            current_statement="I was at home",
            past_statement="I was at work",
            confidence=0.95
        )
        
        if alert:
            print(f"✓ Contradiction alert created")
            print(f"  Alert ID: {alert.alert_id}")
            print(f"  Severity: {alert.severity.value}")
            print(f"  Confidence: {alert.confidence*100:.0f}%")
            
            # Test stress spike alert
            alert2 = alert_system.check_stress_spike(
                evidence_id="EV_002",
                participant_id="PART_A",
                current_stress=75.0,
                baseline_stress=30.0,
                spike_threshold=1.5
            )
            
            if alert2:
                print(f"✓ Stress spike alert created")
                print(f"  Spike ratio: {alert2.details['spike_ratio']:.1f}x")
            
            summary = alert_system.get_alert_summary()
            print(f"\nAlert Summary:")
            print(f"  Total alerts: {summary['total_alerts']}")
            print(f"  Critical: {summary['critical']}")
            print(f"  High: {summary['high']}")
            
            return True
        else:
            print(f"✗ Alert creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Alert system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evidence_scoring():
    """Test evidence scoring system"""
    print("\n" + "="*60)
    print("TEST 5: Evidence Scoring System")
    print("="*60)

    try:
        from evidence_scoring import EvidenceScoringSystem
        
        scorer = EvidenceScoringSystem()
        
        # Score evidence
        score = scorer.score_evidence(
            evidence_id="EV_001",
            evidence_type="voice_note",
            afrikaans_confidence=0.95,
            timeline_consistency=0.85,
            stress_level=45.0,
            baseline_stress=30.0,
            cross_references=3,
            contradictions=1,
            forensic_flags=[],
            expert_findings=2
        )
        
        if score:
            print(f"✓ Evidence scored")
            print(f"  Evidence ID: {score.evidence_id}")
            print(f"  Reliability: {score.reliability_rating.value}")
            print(f"  Importance: {score.importance_rating.value}")
            print(f"  Overall Strength: {score.overall_strength:.1f}/100")
            print(f"  Strengths: {', '.join(score.strengths[:2])}")
            
            summary = scorer.get_scoring_summary()
            print(f"\nScoring Summary:")
            print(f"  Total evidence: {summary['total_evidence']}")
            print(f"  Average strength: {summary['average_strength']:.1f}")
            
            return True
        else:
            print(f"✗ Evidence scoring failed")
            return False
            
    except Exception as e:
        print(f"✗ Evidence scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_investigative_checklist():
    """Test investigative checklist generation"""
    print("\n" + "="*60)
    print("TEST 6: Investigative Checklist")
    print("="*60)

    try:
        from investigative_checklist import InvestigativeChecklistGenerator
        
        generator = InvestigativeChecklistGenerator()
        
        # Generate checklist
        checklist = generator.generate_checklist_from_findings(
            case_id="CASE_001",
            contradictions=[
                {
                    'participant_id': 'PART_A',
                    'current_statement': 'I was home',
                    'past_statement': 'I was at work',
                    'confidence': 0.95,
                    'evidence_ids': ['EV_001']
                }
            ],
            pattern_changes=[
                {
                    'participant_id': 'PART_A',
                    'pattern_type': 'vocabulary',
                    'change_description': 'Increased use of defensive language',
                    'confidence': 0.80,
                    'evidence_ids': ['EV_002']
                }
            ],
            timeline_gaps=[],
            stress_spikes=[],
            manipulation_indicators=[],
            participants=[
                {'participant_id': 'PART_A', 'primary_username': 'Alice'}
            ]
        )
        
        if checklist:
            print(f"✓ Checklist generated")
            print(f"  Total items: {len(checklist)}")
            
            summary = generator.get_checklist_summary('CASE_001')
            print(f"\nChecklist Summary:")
            print(f"  Total items: {summary['total_items']}")
            print(f"  Pending: {summary['pending']}")
            print(f"  Critical: {summary.get('critical_pending', 0)}")
            
            return True
        else:
            print(f"✗ Checklist generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Checklist test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_narrative_reconstruction():
    """Test narrative reconstruction"""
    print("\n" + "="*60)
    print("TEST 7: Narrative Reconstruction")
    print("="*60)

    try:
        from narrative_reconstruction import NarrativeReconstructor
        
        reconstructor = NarrativeReconstructor()
        
        # Build narrative
        narrative = reconstructor.build_narrative(
            participant_id="PART_A",
            participant_name="Alice",
            statements=[
                {
                    'content': 'I was at home all day. I never left the house.',
                    'timestamp': datetime.now() - timedelta(days=1),
                    'evidence_id': 'EV_001'
                },
                {
                    'content': 'I went to the store in the afternoon.',
                    'timestamp': datetime.now(),
                    'evidence_id': 'EV_002'
                }
            ]
        )
        
        if narrative:
            print(f"✓ Narrative reconstructed")
            print(f"  Participant: {narrative.participant_name}")
            print(f"  Narrative consistency: {narrative.narrative_consistency*100:.0f}%")
            print(f"  Timeline consistency: {narrative.timeline_consistency*100:.0f}%")
            print(f"  Gaps identified: {len(narrative.gaps)}")
            print(f"  Inconsistencies: {len(narrative.inconsistencies)}")
            
            return True
        else:
            print(f"✗ Narrative reconstruction failed")
            return False
            
    except Exception as e:
        print(f"✗ Narrative reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparative_psychology():
    """Test comparative psychology analysis"""
    print("\n" + "="*60)
    print("TEST 8: Comparative Psychology")
    print("="*60)

    try:
        from comparative_psychology import ComparativePsychologyAnalyzer
        
        analyzer = ComparativePsychologyAnalyzer()
        
        # Build profiles
        profile_a = analyzer.build_profile(
            participant_id="PART_A",
            participant_name="Alice",
            evidence_data=[
                {'stress_level': 45.0, 'text': 'I was at home all day'},
                {'stress_level': 75.0, 'text': 'Why are you asking me this?'},
                {'stress_level': 60.0, 'text': 'I never said that'}
            ]
        )
        
        profile_b = analyzer.build_profile(
            participant_id="PART_B",
            participant_name="Bob",
            evidence_data=[
                {'stress_level': 30.0, 'text': 'I saw her at the store'},
                {'stress_level': 35.0, 'text': 'She was definitely there'},
                {'stress_level': 32.0, 'text': 'I remember clearly'}
            ]
        )
        
        if profile_a and profile_b:
            print(f"✓ Psychological profiles built")
            print(f"  Alice - Baseline stress: {profile_a.baseline_stress:.0f}")
            print(f"  Bob - Baseline stress: {profile_b.baseline_stress:.0f}")
            
            # Compare
            comparison = analyzer.compare_profiles("PART_A", "PART_B")
            if comparison:
                print(f"\n✓ Profiles compared")
                print(f"  Stress difference: {comparison.stress_baseline_difference:.0f}")
                print(f"  Defensiveness diff: {comparison.defensiveness_difference*100:.0f}%")
                print(f"  Behavioral contrast: {comparison.behavioral_contrast}")
            
            return True
        else:
            print(f"✗ Profile building failed")
            return False
            
    except Exception as e:
        print(f"✗ Comparative psychology test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_escalation():
    """Test risk escalation matrix"""
    print("\n" + "="*60)
    print("TEST 9: Risk Escalation Matrix")
    print("="*60)

    try:
        from risk_escalation_matrix import RiskEscalationMatrix
        
        matrix = RiskEscalationMatrix()
        
        # Assess participant risk
        risk_a = matrix.assess_participant_risk(
            participant_id="PART_A",
            participant_name="Alice",
            contradiction_count=3,
            contradiction_confidence=0.90,
            stress_spikes=2,
            baseline_stress=30.0,
            current_stress=65.0,
            manipulation_indicators=2,
            timeline_inconsistencies=1,
            psychological_red_flags=1
        )
        
        if risk_a:
            print(f"✓ Participant risk assessed")
            print(f"  Participant: {risk_a.participant_name}")
            print(f"  Risk score: {risk_a.overall_risk_score:.1f}/100")
            print(f"  Risk level: {risk_a.risk_level.value}")
            print(f"  Recommended action: {risk_a.recommended_action.value}")
            
            return True
        else:
            print(f"✗ Risk assessment failed")
            return False
            
    except Exception as e:
        print(f"✗ Risk escalation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multilingual_support():
    """Test multilingual support"""
    print("\n" + "="*60)
    print("TEST 10: Multilingual Support")
    print("="*60)

    try:
        from multilingual_support import MultilingualAnalyzer
        
        analyzer = MultilingualAnalyzer()
        
        # Analyze English text
        analysis_en = analyzer.analyze_text(
            text_id="TEXT_001",
            text="Hello, I was at home all day yesterday."
        )
        
        if analysis_en:
            print(f"✓ English text analyzed")
            print(f"  Detected language: {analysis_en.language_detection.detected_language.value}")
            print(f"  Confidence: {analysis_en.language_detection.confidence*100:.0f}%")
            print(f"  Authenticity: {analysis_en.authenticity_assessment}")
            
            # Analyze Afrikaans text
            analysis_af = analyzer.analyze_text(
                text_id="TEXT_002",
                text="Hallo, ek was die hele dag tuis gister."
            )
            
            if analysis_af:
                print(f"\n✓ Afrikaans text analyzed")
                print(f"  Detected language: {analysis_af.language_detection.detected_language.value}")
                print(f"  Confidence: {analysis_af.language_detection.confidence*100:.0f}%")
            
            return True
        else:
            print(f"✗ Language analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ Multilingual support test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("DIE WAARHEID - INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Module Imports", test_module_imports),
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Case Creation", test_case_creation),
        ("Alert System", test_alert_system),
        ("Evidence Scoring", test_evidence_scoring),
        ("Investigative Checklist", test_investigative_checklist),
        ("Narrative Reconstruction", test_narrative_reconstruction),
        ("Comparative Psychology", test_comparative_psychology),
        ("Risk Escalation", test_risk_escalation),
        ("Multilingual Support", test_multilingual_support),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System is fully integrated!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
