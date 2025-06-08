#!/bin/bash
# run_unit_tests.sh - Run unit tests without Redis dependency

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path to include src directory
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}:${PYTHONPATH}"

echo "Running unit tests for valkey_stress_test..."
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# Create necessary __init__.py files
touch "${SCRIPT_DIR}/tests/__init__.py"
touch "${SCRIPT_DIR}/tests/unit/__init__.py"
mkdir -p "${SCRIPT_DIR}/tests/integration"
touch "${SCRIPT_DIR}/tests/integration/__init__.py"

# Run unit tests
python -m pytest tests/unit/test_core_components.py tests/unit/test_workload_components.py -v --tb=short "$@"