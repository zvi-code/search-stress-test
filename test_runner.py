#!/usr/bin/env python3
"""
Test runner that properly configures the Python path and runs pytest.

Usage:
    python test_runner.py              # Run all unit tests
    python test_runner.py -v           # Verbose output
    python test_runner.py -k test_name # Run specific test
"""

import sys
import os
from pathlib import Path
import subprocess

# Setup Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

# Add paths to PYTHONPATH
paths_to_add = [str(src_path), str(project_root)]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Also set PYTHONPATH environment variable for subprocess
env = os.environ.copy()
env['PYTHONPATH'] = os.pathsep.join(paths_to_add + [env.get('PYTHONPATH', '')])

# Create necessary __init__.py files
init_files = [
    project_root / "tests" / "__init__.py",
    project_root / "tests" / "unit" / "__init__.py",
    project_root / "tests" / "integration" / "__init__.py",
]

for init_file in init_files:
    init_file.parent.mkdir(parents=True, exist_ok=True)
    if not init_file.exists():
        init_file.touch()

# Prepare pytest command
cmd = [sys.executable, "-m", "pytest", "-m", "unit"]

# Add any additional arguments passed to this script
if len(sys.argv) > 1:
    cmd.extend(sys.argv[1:])
else:
    # Default to verbose output
    cmd.append("-v")

print(f"Running command: {' '.join(cmd)}")
print(f"Python path: {sys.path[:2]}")
print("-" * 80)

# Run pytest
result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)