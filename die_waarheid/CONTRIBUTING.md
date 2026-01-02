# Contributing to Die Waarheid

Thank you for your interest in contributing to Die Waarheid! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please review our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (venv or conda)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/die-waarheid.git
cd die-waarheid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 mypy
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow PEP 8 style guide
- Add type hints to functions
- Write docstrings for classes and methods
- Add tests for new functionality

### 3. Run Tests
```bash
# Run integration tests
python test_integration.py

# Run linting
flake8 src/

# Run type checking
mypy src/

# Format code
black src/
```

### 4. Commit Changes
```bash
git add .
git commit -m "Brief description of changes

- Detailed explanation of what changed
- Why the change was made
- Any related issues or PRs"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title
- Description of changes
- Reference to any related issues
- Screenshots/examples if applicable

## Coding Standards

### Style Guide
- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable names

### Type Hints
```python
def analyze_statement(text: str, speaker_id: str) -> Dict[str, Any]:
    """Analyze a statement for forensic indicators."""
    pass
```

### Docstrings
```python
def calculate_risk_score(factors: List[float]) -> float:
    """
    Calculate overall risk score from multiple factors.
    
    Args:
        factors: List of risk factors (0-1 scale)
    
    Returns:
        Overall risk score (0-100)
    
    Raises:
        ValueError: If factors contain invalid values
    """
    pass
```

### Comments
- Use comments to explain WHY, not WHAT
- Keep comments concise and clear
- Update comments when code changes

## Testing

### Write Tests
```python
def test_alert_system_detects_contradictions():
    """Test that alert system detects statement contradictions."""
    alert_system = AlertSystem()
    result = alert_system.check_contradiction(statement1, statement2)
    assert result.contradiction_detected == True
```

### Test Coverage
- Aim for 80%+ code coverage
- Test edge cases and error conditions
- Test integration between modules

### Run Tests
```bash
# Run all tests
python test_integration.py

# Run with coverage
pytest --cov=src test_integration.py
```

## Documentation

### Update Documentation
- Update README.md if adding features
- Add docstrings to new functions
- Update SYSTEM_COMPLETE.md for major changes
- Add examples for new modules

### Documentation Standards
- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep documentation up-to-date

## Commit Message Guidelines

### Format
```
<type>: <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions/changes
- `chore`: Build/dependency changes

### Example
```
feat: Add sentiment analysis to text forensics

- Implement sentiment scoring for statements
- Add emotional intensity tracking
- Integrate with narrative reconstruction

Closes #123
```

## Pull Request Process

1. **Create PR** with clear title and description
2. **Pass Tests** - All tests must pass
3. **Code Review** - Wait for review from maintainers
4. **Address Feedback** - Make requested changes
5. **Merge** - PR will be merged after approval

## Reporting Issues

### Bug Reports
Include:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and OS
- Error messages/logs

### Feature Requests
Include:
- Clear description of feature
- Use case/motivation
- Proposed implementation (optional)
- Examples (optional)

## Questions?

- **Documentation**: Check [SYSTEM_COMPLETE.md](SYSTEM_COMPLETE.md)
- **Issues**: Search existing [GitHub Issues](https://github.com/YOUR_USERNAME/die-waarheid/issues)
- **Discussions**: Start a [GitHub Discussion](https://github.com/YOUR_USERNAME/die-waarheid/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Die Waarheid!
