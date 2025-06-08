#!/bin/bash
# Run the unit tests after applying fixes

echo "Running unit tests with fixes applied..."
echo "========================================"

# Set Python path
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"

# Run tests without coverage to see results clearly
python -m pytest tests/unit/ -v --tb=short -x

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "To run with coverage:"
    echo "  pytest -m unit --cov=valkey_stress_test --cov-report=html"
else
    echo ""
    echo "❌ Some tests failed. See output above."
fi