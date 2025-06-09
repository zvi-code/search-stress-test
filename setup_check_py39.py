#!/usr/bin/env python3
"""
Quick setup script for Valkey Stress Test tool (Python 3.9+ compatible).

This script checks your system and provides guidance for installation.
"""

import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✓ Python version is compatible (3.9+ required)")
        return True
    else:
        print("✗ Python 3.9+ required")
        print("\nSolutions:")
        print("1. Ubuntu/Debian: sudo apt install python3.9")
        print("2. Use pyenv: curl https://pyenv.run | bash && pyenv install 3.9.21")
        print("3. Use conda: conda create -n valkey-stress python=3.9")
        print("4. Use Docker: docker run -it python:3.9-slim")
        return False

def check_git():
    """Check if git is available."""
    if shutil.which("git"):
        print("✓ Git is available")
        return True
    else:
        print("✗ Git not found")
        print("Install git: sudo apt install git (Ubuntu) or brew install git (macOS)")
        return False

def check_pip():
    """Check if pip is available."""
    if shutil.which("pip") or shutil.which("pip3"):
        print("✓ pip is available")
        return True
    else:
        print("✗ pip not found")
        print("Install pip: sudo apt install python3-pip")
        return False

def check_requirements_file():
    """Check if requirements.txt exists."""
    if Path("requirements_py39.txt").exists():
        print("✓ Python 3.9 compatible requirements file found")
        return True
    elif Path("requirements.txt").exists():
        print("✓ Standard requirements file found")
        print("⚠️  Using Python 3.9 compatible dependency versions")
        return True
    else:
        print("✗ Requirements file not found")
        return False

def main():
    """Main setup check function."""
    print("🔍 Checking system requirements for Valkey Stress Test (Python 3.9+ compatible)...")
    print()
    
    all_good = True
    
    # Check Python version
    all_good &= check_python_version()
    print()
    
    # Check git
    all_good &= check_git()
    print()
    
    # Check pip
    all_good &= check_pip()
    print()
    
    # Check requirements file
    all_good &= check_requirements_file()
    print()
    
    if all_good:
        print("🎉 All requirements met! You can proceed with installation.")
        print()
        print("Next steps:")
        print("1. git clone https://github.com/your-org/valkey_stress_test.git")
        print("2. cd valkey_stress_test")
        
        # Check if we're on Amazon Linux for special script
        try:
            with open("/etc/os-release") as f:
                os_info = f.read()
            if "Amazon Linux" in os_info:
                print("3. ./setup_ec2_py39.sh  # Special Amazon Linux script")
            else:
                print("3. pip install -r requirements_py39.txt  # Python 3.9 compatible")
                print("4. pip install -e .")
        except:
            print("3. pip install -r requirements_py39.txt")
            print("4. pip install -e .")
        
        print("5. python verify_installation.py")
    else:
        print("❌ Some requirements are missing. Please install them before proceeding.")
        print()
        print("For AWS EC2 Amazon Linux, you can use:")
        print("./setup_ec2_py39.sh")

if __name__ == "__main__":
    main()
