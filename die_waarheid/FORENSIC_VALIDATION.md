# Forensic Validation and Reliability Guide

## Overview

Die Waarheid is designed as a forensic analysis tool. This document outlines the validation procedures, reliability measures, and audit trails necessary to ensure the system produces trustworthy, reproducible results suitable for legal and investigative use.

## Core Principles

### 1. Chain of Custody
- Every file processed is hashed (SHA-256) and tracked
- Original paths and timestamps are preserved
- All processing occurs on copies in secured case folders
- Full audit trail maintained for each analysis run

### 2. Provenance and Reproducibility
- Configuration snapshots saved with each run
- Model versions and parameters recorded
- Random seeds documented where applicable
- Same input + config = same output (deterministic)

### 3. No Placeholders Allowed
- All outputs must come from actual processing
- Failed stages marked as "FAILED" - never filled with dummy data
- Partial results clearly labeled as incomplete
- Human review required for any high-impact findings

## Validation Procedures

### A. Verification Harness
Run before trusting any analysis:

```bash
# Verify pipeline on sample files
python verification_harness.py --files file1.wav file2.wav --output data/verification
```

The harness produces:
- Input manifest with SHA-256 hashes
- Configuration snapshot
- Processing logs
- Stage-by-stage results
- Summary report

### B. Known Phrase Testing
For Afrikaans verification:
1. Select 5-10 files with known Afrikaans phrases
2. Run verification harness
3. Verify key phrases appear in transcripts
4. Check Afrikaans processing confidence scores

### C. Regression Tests
Run automated test suite:

```bash
python automated_tests.py
```

Tests cover:
- Afrikaans text processing
- Code-switch detection
- Speaker diarization
- Background sound analysis
- Integration and reproducibility

## Audit Trail Requirements

### For Each Case Run
The system creates a run folder containing:
```
data/cases/<case_id>/runs/<timestamp>/
├── inputs.json          # File list with hashes
├── config.json          # Model versions and settings
├── process.log          # Complete processing log
├── results.json         # Full analysis results
├── summary.json         # Executive summary
└── intermediate/        # Optional intermediate outputs
```

### For Each File
- Original path and hash
- Processing timestamps
- Stage-by-stage results
- Any errors or warnings
- Final output with confidence scores

## Language Processing Validation

### Afrikaans Primary Language
System prioritizes Afrikaans with:
- Whisper model fine-tuned for Afrikaans
- Triple-check verification process
- Afrikaans-specific normalization
- Code-switch detection for English

### Validation Steps
1. **Language Detection**: Verify correct identification of Afrikaans vs English
2. **Text Verification**: Confirm triple-check process improves accuracy
3. **Code-Switch Detection**: Test on mixed-language samples
4. **Cultural Context**: Ensure idioms and colloquialisms preserved

### Background Sound Analysis
System identifies and classifies:
- Traffic noise
- Machinery sounds
- Weather (wind, rain)
- Crowd noise
- Music
- Silence

Validation requires:
- Test files with known background conditions
- Verification of sound type classification
- Confidence scoring calibration

## Risk Mitigation

### Human Oversight
- All "risk", "contradiction", or "credibility" scores marked as "assistive"
- Require human review for any legal decisions
- Clear citations to source material
- Never present AI analysis as definitive truth

### Error Handling
- Transparent error reporting
- Graceful degradation (partial results)
- Clear indication of failed stages
- Recovery procedures for interrupted runs

### Data Protection
- No temporary storage of evidence
- Encrypted storage options available
- Access logging and audit trails
- Secure data retention policies

## Quality Assurance

### Automated Checks
- File integrity verification
- Configuration validation
- Output format compliance
- No placeholder detection

### Manual Review Points
- Sample transcript verification
- Speaker label accuracy
- Background sound classification
- Contradiction detection validity

### Continuous Improvement
- Collect feedback on accuracy
- Update models based on errors
- Expand test case library
- Refine confidence thresholds

## Usage Guidelines

### Before Analysis
1. Run verification harness on sample files
2. Confirm Afrikaans phrases are captured correctly
3. Validate speaker profiles if used
4. Check background sound detection

### During Analysis
1. Monitor progress logs
2. Watch for error messages
3. Verify file counts match expectations
4. Check confidence scores

### After Analysis
1. Review summary report
2. Spot-check key transcripts
3. Verify speaker assignments
4. Cross-reference contradictions

### For Legal Use
1. Document the verification process
2. Preserve all run artifacts
3. Prepare expert witness testimony
4. Address potential challenges to AI analysis

## Troubleshooting

### Common Issues
- **Low confidence scores**: Check audio quality, verify Afrikaans clarity
- **Missing Afrikaans phrases**: Verify language detection settings
- **Incorrect speaker labels**: Review speaker training samples
- **Background noise errors**: Check audio separation parameters

### Validation Failures
If verification harness fails:
1. Check input file integrity
2. Verify configuration settings
3. Review error logs
4. Run individual module tests
5. Contact support with full artifacts

## Compliance

### Standards Compliance
- ISO 17025 (Testing and calibration)
- ISO 9001 (Quality management)
- ASCLD/LAB accreditation requirements
- Local forensic science standards

### Legal Admissibility
- Daubert standard compliance
- Frye test considerations
- Expert witness qualifications
- Method validation documentation

## Conclusion

Die Waarheid provides forensic-grade analysis when used according to these guidelines. The combination of automated verification, human oversight, and comprehensive audit trails ensures results that can withstand legal scrutiny and investigative review.

Remember: AI analysis is a tool to assist human experts, not replace them. Always verify critical findings and maintain professional skepticism.
