#!/usr/bin/env python3
"""
Verify that the test fixes work correctly.
This script runs a subset of the previously failing tests to verify they now pass.
"""

import subprocess
import sys

def run_specific_test(test_path):
    """Run a specific test and return whether it passed."""
    cmd = [sys.executable, "-m", "pytest", test_path, "-v", "-x"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def main():
    """Run the previously failing tests to verify fixes."""
    failing_tests = [
        "tests/unit/test_workload_components.py::TestBatchedWorkload::test_batch_success_recording",
        "tests/unit/test_workload_components.py::TestRateLimitedWorkload::test_rate_limiting",
        "tests/unit/test_workload_components.py::TestQueryWorkload::test_query_execution",
        "tests/unit/test_workload_components.py::TestShrinkWorkload::test_deletion_execution",
        "tests/unit/test_workload_components.py::TestWorkloadRegistry::test_workload_registration",
        "tests/unit/test_workload_components.py::TestWorkloadIntegration::test_full_workload_lifecycle",
    ]
    
    print("Verifying test fixes...\n")
    
    all_passed = True
    for test in failing_tests:
        print(f"Running: {test}")
        passed, stdout, stderr = run_specific_test(test)
        
        if passed:
            print("✅ PASSED\n")
        else:
            print("❌ FAILED")
            print("STDOUT:", stdout[-500:])  # Last 500 chars
            print("STDERR:", stderr[-500:])
            print()
            all_passed = False
    
    if all_passed:
        print("\n🎉 All previously failing tests now pass!")
        return 0
    else:
        print("\n❌ Some tests are still failing. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())