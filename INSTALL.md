# Installation Guide

This guide provides step-by-step instructions for installing the Valkey Stress Test tool.

## Prerequisites

- **Python 3.10 or higher** - Check with `python3 --version`
- **Git** - For cloning the repository
- **Redis/Valkey instance** - With Search module enabled

## Quick Installation (Recommended)

### Option 1: Using pip (Simplest)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in development mode
pip install -e .

# 4. Verify installation
python verify_installation.py
```

### Option 2: AWS EC2 / Cloud Server Installation

For AWS EC2 or other cloud servers:

```bash
# 1. Update system packages
sudo yum update -y  # Amazon Linux
# or: sudo apt update && sudo apt upgrade -y  # Ubuntu

# 2. Install Python 3.10+ and Git
sudo yum install python3 python3-pip git -y  # Amazon Linux
# or: sudo apt install python3 python3-pip python3-venv git -y  # Ubuntu

# 3. Create working directory
mkdir -p ~/valkey-stress
cd ~/valkey-stress

# 4. Clone and install
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# 5. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 6. Install dependencies and package
pip install -r requirements.txt
pip install -e .

# 7. Verify installation
vst --help
```

### Option 3: Using Poetry (For Developers)

```bash
# 1. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone and setup
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# 3. Install dependencies and package
poetry install

# 4. Activate the environment
poetry shell

# 5. Verify installation
vst --help
```

## Detailed Installation Steps

### Step 1: System Requirements

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

**CentOS/RHEL:**
```bash
sudo yum install python3 python3-pip git
# or for newer versions:
sudo dnf install python3 python3-pip git
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git
```

### Step 2: Clone Repository

```bash
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install visualization dependencies
pip install -r requirements-viz.txt

# For developers: Install development dependencies
pip install -r requirements-dev.txt
```

### Step 5: Install the Package

```bash
# Install in development mode (recommended)
pip install -e .

# This makes the 'vst' command available globally
```

### Step 6: Verify Installation

```bash
# Check if vst command is available
vst --help

# Check system information
vst info system

# Run comprehensive verification
python verify_installation.py

# Check Python environment
which vst
python -c "import valkey_stress_test; print(valkey_stress_test.__file__)"
```

## Setting Up Redis/Valkey

### Using Docker (Recommended)

```bash
# Redis Stack (includes Search module)
docker run -d --name redis-stack \
  -p 6379:6379 \
  redis/redis-stack-server:latest

# Verify it's working
vst info redis
```

### Manual Installation

**Install Redis Stack:**
- Follow [Redis Stack installation guide](https://redis.io/docs/stack/get-started/install/)

**Or install Valkey with Search:**
- Follow [Valkey installation guide](https://valkey.io/docs/intro/)

## Troubleshooting

### Common Issues

**1. "vst: command not found"**

```bash
# Check if package is installed
pip list | grep valkey

# If not installed, run:
pip install -e .

# If still not working, check PATH
echo $PATH
which python
```

**2. "Module not found" errors**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**3. Permission errors**

```bash
# Use --user flag
pip install --user -r requirements.txt
pip install --user -e .

# Or use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**4. Python version issues**

```bash
# Check Python version
python3 --version

# If < 3.10, install newer Python:
# Ubuntu:
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv

# CentOS/RHEL: Install from EPEL or compile from source
# macOS: Use Homebrew or pyenv
```

**5. Redis connection issues**

```bash
# Check if Redis is running
redis-cli ping

# Start Redis with Docker
docker run -d -p 6379:6379 redis/redis-stack-server:latest

# Check connection with vst
vst info redis --host localhost --port 6379
```

### AWS EC2 Specific Issues

**1. Python version on Amazon Linux**

```bash
# Check Python version
python3 --version

# If Python < 3.10, install newer version:
sudo amazon-linux-extras install python3.8
# or compile Python 3.10+ from source
```

**2. Package not found on EC2**

```bash
# The package is not on PyPI yet - must install from source
# Follow the EC2 installation steps above

# Make sure you're cloning the repository:
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test
pip install -r requirements.txt
pip install -e .
```

**3. Poetry not available**

```bash
# Poetry is not required! Use pip instead:
pip install -r requirements.txt
pip install -e .

# If you want Poetry:
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

**4. Virtual environment issues on EC2**

```bash
# Create virtual environment manually
python3 -m venv ~/venv
source ~/venv/bin/activate

# Verify virtual environment is active
which python
which pip

# Install in virtual environment
pip install -r requirements.txt
pip install -e .
```

### Getting Help

**Check installation:**
```bash
vst info system
vst info redis
```

**Enable debug mode:**
```bash
vst --verbose info system
```

**Run diagnostics:**
```bash
# Test basic functionality
vst run quick --duration 30 --dry-run
```

## Next Steps

After successful installation:

1. **Quick Test**: `vst run quick --duration 60`
2. **Read Documentation**: Check `docs/GETTING_STARTED.md`
3. **Download Datasets**: `vst dataset list`
4. **Create Scenarios**: Copy from `config/scenarios/`

## Development Installation

For contributing to the project:

```bash
# Clone with development setup
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# Install with Poetry
poetry install --with dev

# Or with pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest
# or
python run_tests.py

# Format code
black .
ruff check .
```

## Alternative Installation Methods

### Using conda

```bash
# Create conda environment
conda create -n valkey-stress python=3.10
conda activate valkey-stress

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### System-wide installation (not recommended)

```bash
# Clone repository
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# Install system-wide
sudo pip install -r requirements.txt
sudo pip install -e .
```

## Uninstalling

```bash
# Uninstall the package
pip uninstall valkey-stress-test

# Remove virtual environment (if used)
rm -rf venv

# Remove cloned repository
cd ..
rm -rf valkey_stress_test
```

---

For more help, see:
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Troubleshooting](docs/README.md#troubleshooting)
