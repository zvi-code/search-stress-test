#!/usr/bin/env python3
"""
Setup test environment and fix import paths.

This script ensures that the valkey_stress_test package can be imported
during testing, even if it's not installed via pip.
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent

# Add src directory to Python path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add the project root for tests to find the mocks
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Python path configured:")
print(f"  - Project root: {project_root}")
print(f"  - Source path: {src_path}")
print(f"  - Python version: {sys.version}")

# Create empty __init__.py files if they don't exist
init_files = [
    project_root / "tests" / "__init__.py",
    project_root / "tests" / "unit" / "__init__.py",
    project_root / "tests" / "integration" / "__init__.py",
]

for init_file in init_files:
    if not init_file.exists():
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.touch()
        print(f"Created: {init_file}")

# Try importing the package
try:
    import valkey_stress_test
    print("\n✅ Successfully imported valkey_stress_test package")
except ImportError as e:
    print(f"\n❌ Failed to import package: {e}")
    sys.exit(1)

# Try importing test utilities
try:
    from tests.mocks import MockRedisClient
    print("✅ Successfully imported test mocks")
except ImportError as e:
    print(f"❌ Failed to import test mocks: {e}")
    sys.exit(1)

print("\nTest environment setup complete! You can now run:")
print("  pytest -m unit")
print("  pytest -m unit -v  # verbose output")
print("  pytest -m unit -s  # show print statements")