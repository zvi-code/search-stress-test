# Quick Start Guide for Testing

## The Issue
When you run `pytest -m unit`, no tests are collected because:
1. The package isn't installed, so imports fail
2. Python can't find the `valkey_stress_test` module
3. The test markers weren't properly applied

## Solution: Install in Development Mode

### Step 1: Install the package in development mode
```bash
pip install -e .
```

Or with test dependencies:
```bash
pip install -e ".[test]"
```

### Step 2: Run the tests
Now pytest will work properly:
```bash
# Run all unit tests
pytest -m unit

# Run with verbose output
pytest -m unit -v

# Run a specific test file
pytest tests/test_simple.py -v

# Run with coverage
pytest -m unit --cov=valkey_stress_test
```

## Alternative: Use the Test Runner
If you don't want to install the package:

```bash
# Make the runner executable
chmod +x test_runner.py

# Run tests
python test_runner.py
```

## Alternative: Set PYTHONPATH manually
```bash
# From project root
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
pytest -m unit -v
```

## What the Tests Do
- **No Redis Required**: All tests use mocks
- **Fast Execution**: Tests run in seconds
- **Comprehensive Coverage**: Tests cover core logic without external dependencies

## Test Structure
```
tests/
├── test_simple.py              # Basic tests to verify setup
├── unit/
│   ├── test_core_components.py # Tests for vector operations, metrics, etc.
│   └── test_workload_components.py # Tests for workload execution
└── mocks.py                    # Mock Redis client and helpers
```

## Common Commands
```bash
# Install for development
pip install -e ".[test]"

# Run all unit tests
pytest -m unit

# Run with output
pytest -m unit -s

# Run specific test
pytest -k "test_vector_operations"

# Generate coverage report
pytest -m unit --cov=valkey_stress_test --cov-report=html
```

## Troubleshooting

### Import Errors
If you see import errors, ensure you've installed the package:
```bash
pip install -e .
```

### No Tests Collected
Make sure you're in the project root directory and the package is installed.

### Asyncio Warnings
The pytest.ini file includes `asyncio_default_fixture_loop_scope = function` to fix asyncio warnings.

## Success!
When tests run successfully, you'll see:
```
================================================================ test session starts ================================================================
platform darwin -- Python 3.12.3, pytest-8.3.4, pluggy-1.5.0
collected 30 items

tests/unit/test_core_components.py::TestVectorOperations::test_calculate_norm_single_vector PASSED
tests/unit/test_core_components.py::TestVectorOperations::test_calculate_norm_batch PASSED
...
================================================================ 30 passed in 2.34s =================================================================
```