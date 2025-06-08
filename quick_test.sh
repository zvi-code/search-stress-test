#!/bin/bash
# Quick test runner - runs unit tests with minimal output

echo "Running unit tests..."
echo "===================="

# Use minimal conftest if it exists
if [ -f "tests/conftest_minimal.py" ]; then
    echo "Using minimal conftest.py to avoid warnings..."
    cp tests/conftest_minimal.py tests/conftest.py
fi

# Run tests with warnings ignored
python -m pytest -m unit -v -W ignore::DeprecationWarning --tb=short

# Show summary
echo ""
echo "Test run complete!"
echo ""
echo "To see more details, run:"
echo "  pytest -m unit -v"
echo ""
echo "To generate coverage report:"
echo "  pytest -m unit --cov=valkey_stress_test --cov-report=html"