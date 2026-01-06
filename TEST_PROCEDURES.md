# Test Procedures Documentation

## Overview

This document outlines the testing procedures for the Die Waarheid forensic analysis system.

## Test Structure

### Unit Tests

Located in `tests/unit/`:

- `test_pipeline_processor.py` - Tests for the main pipeline orchestration
- `test_ai_analyzer.py` - Tests for AI analysis and caching
- `test_forensics.py` - Tests for audio forensic analysis

### Integration Tests

Located in `tests/integration/`:

- `test_pipeline_integration.py` - End-to-end pipeline testing

### Test Fixtures

Located in `tests/fixtures/`:

- Mock data and test audio files
- Shared test configurations

## Running Tests

### Install Test Dependencies

```bash
pip install -r test-requirements.txt
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=die_waarheid --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Tests with specific markers
pytest tests/ -m "unit"
pytest tests/ -m "integration"
```

## Test Categories and Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, multiple components)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.api` - Tests requiring API keys

## Known Test Issues

1. **ForensicsEngine Method Names**: Some test methods use different naming conventions than the actual implementation
2. **Mock Configuration**: Some mocks need proper setup for context managers
3. **File Path Handling**: Tests need to create actual files for path validation
4. **Floating Point Precision**: Risk score calculations need proper assertion tolerance

## Test Coverage Goals

- Target: 80% code coverage
- Critical paths: 95% coverage
- Error handling: 100% coverage

## Continuous Integration

Tests should run automatically on:

- Pull requests
- Main branch commits
- Release candidates

## Test Data Management

- Use temporary directories for test files
- Clean up test data after each test
- Use fixtures for consistent test data
- Mock external APIs to avoid rate limits

## Debugging Failed Tests

1. Run with verbose output: `pytest -v`
2. Run with debugging: `pytest --pdb`
3. Run specific test: `pytest tests/unit/test_file.py::TestClass::test_method`
4. Use print statements or logging for debugging

## Best Practices

1. Keep tests independent and isolated
2. Use descriptive test names
3. Mock external dependencies
4. Test both success and failure scenarios
5. Use fixtures for common test setup
6. Clean up resources after tests
