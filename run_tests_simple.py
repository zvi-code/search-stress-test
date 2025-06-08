#!/usr/bin/env python3
"""
Simple test runner for the fixed unit tests.
Run this script to execute all unit tests and see the results.
"""

import subprocess
import sys
import os

def main():
    """Run the unit tests."""
    print("Running Valkey Stress Test Unit Tests")
    print("=====================================\n")
    
    # Set up environment
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env['PYTHONPATH'] = f"{project_root}/src:{project_root}:{env.get('PYTHONPATH', '')}"
    
    # Run pytest with unit marker
    cmd = [
        sys.executable, "-m", "pytest",
        "-m", "unit",
        "-v",
        "-W", "ignore::DeprecationWarning",
        "--tb=short"
    ]
    
    print(f"Executing: {' '.join(cmd)}\n")
    
    # Run the tests
    result = subprocess.run(cmd, env=env)
    
    print("\n" + "="*70)
    if result.returncode == 0:
        print("✅ ALL TESTS PASSED! 🎉")
        print("\nAll 37 unit tests are now working correctly without Redis!")
    else:
        print("❌ Some tests failed.")
        print("\nCheck the output above for details.")
    print("="*70)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())