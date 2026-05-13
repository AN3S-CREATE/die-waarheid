# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-05-13

### Fixed
- **Critical**: Fixed "Status code 410 is not ok" error from deprecated Gemini API model
  - Changed default model from `gemini-2.0-flash` to `gemini-1.5-flash`
  - Updated `google-generativeai` library from 0.3.2 to 0.8.3
  - Added comprehensive error handling for 410 (Gone/Deprecated) status codes
  - Added error handling for 429 (Quota Exceeded) status codes
  - Implemented graceful fallback to pattern-based analysis when API fails

### Added
- Environment variable `GEMINI_MODEL` for configurable model selection
- Environment variable `USE_FREE_AI` to toggle between Gemini and free local AI
- Environment variable `FREE_AI_DEVICE` for free AI device selection
- Comprehensive error detection and categorization across all AI modules:
  - `die_waarheid/src/ai_analyzer.py`
  - `die_waarheid/src/text_forensics.py`
  - `die_waarheid/src/expert_panel.py`
  - `die_waarheid/src/afrikaans_processor.py`
  - `die_waarheid/src/afrikaans_fallback.py`
- Error type field in API responses for better error categorization
- Helpful warning messages when deprecated models are detected
- Documentation files:
  - `BLACKBOX_CLI_FIX.md` - Technical details of the fix
  - `QUICK_START.md` - User guide and troubleshooting
  - `STATUS_410_FIX_SUMMARY.md` - Complete fix summary
  - `validate_fix.py` - Automated validation script
  - `CHANGELOG.md` - This changelog

### Changed
- Default Gemini model from `gemini-2.0-flash` to `gemini-1.5-flash` (stable)
- Model configuration now reads from environment variable with fallback default
- Error handling now provides specific error types and helpful messages
- Updated `.env.example` with new configuration options
- Enhanced `README.md` with warning about 410 error and link to fix

### Improved
- System reliability with automatic fallback mechanisms
- Error messages now include actionable suggestions
- Configuration flexibility through environment variables
- User experience with clearer error reporting
- Documentation completeness and clarity
- Code maintainability with consistent error handling patterns

### Technical Details
**Files Modified**: 13 total
- Core code: 7 files
- Configuration: 2 files  
- Documentation: 4 files

**Breaking Changes**: None - all changes are backward compatible

**Migration Required**: Optional - update `.env` file for better control

**Dependencies Updated**:
- `google-generativeai`: 0.3.2 → 0.8.3

### Security
- No security-related changes in this release

### Deprecated
- Model `gemini-2.0-flash` is deprecated and should not be used (causes 410 errors)

### Removed
- No features removed in this release

---

## [1.0.0] - 2026-01-06

### Initial Release
- Audio forensics with bio-signal detection
- WhatsApp chat parsing and analysis
- AI-powered psychological profiling
- Speaker identification system
- Multi-format report generation
- FastAPI backend server
- Streamlit web interface
- React frontend
- Security features (authentication, rate limiting)
- Performance optimizations (caching, async operations)

---

## Version History

- **1.0.1** (2026-05-13) - Status code 410 error fix
- **1.0.0** (2026-01-06) - Initial release

---

## How to Update

To update to version 1.0.1:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Verify the fix
python3 validate_fix.py

# Run the application
python3 die_waarheid/launcher.py
```

---

## Support

For issues, questions, or contributions:
- Check documentation: `QUICK_START.md`, `BLACKBOX_CLI_FIX.md`
- Run validation: `python3 validate_fix.py`
- Review logs: `die_waarheid/logs/die_waarheid.log`

---

*Last Updated: 2026-05-13*
