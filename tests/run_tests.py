#!/usr/bin/env python3
"""
Test runner script for valkey_stress_test.

This script demonstrates how to run tests without requiring a Redis/Valkey instance.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list) -> int:
    """Run a command and return the exit code."""
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 80)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run tests for valkey_stress_test")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "quick"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add verbose flag
    if args.verbose:
        cmd.append("-vv")
    
    # Add failfast flag
    if args.failfast:
        cmd.append("-x")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=valkey_stress_test",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add test selection
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
        print("Running unit tests only...")
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
        print("Running integration tests only...")
    elif args.type == "quick":
        cmd.extend(["-m", "not slow"])
        print("Running quick tests (excluding slow tests)...")
    else:
        print("Running all tests...")
    
    # Run the tests
    exit_code = run_command(cmd)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())