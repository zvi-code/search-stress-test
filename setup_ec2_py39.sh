#!/bin/bash

# Amazon Linux EC2 Setup Script for Valkey Stress Test (Python 3.9+ Compatible)
# This script installs Python 3.9+ and the valkey_stress_test package

set -e

echo "🚀 Setting up Valkey Stress Test on Amazon Linux EC2 (Python 3.9+ compatible)..."

# Check if we're on Amazon Linux
if ! grep -q "Amazon Linux" /etc/os-release 2>/dev/null; then
    echo "⚠️  This script is designed for Amazon Linux. Use INSTALL.md for other systems."
    exit 1
fi

# Check current Python version
current_python=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "not found")
echo "Current Python version: $current_python"

# Function to check if Python version is adequate (3.9+)
check_python_version() {
    python3 -c "
import sys
if sys.version_info >= (3, 9):
    print('✓ Python 3.9+ available')
    exit(0)
else:
    print('❌ Python 3.9+ required')
    exit(1)
" 2>/dev/null
}

# Check if Python 3.9+ is already available
if check_python_version; then
    echo "✓ Python 3.9+ already available"
    python_cmd="python3"
else
    echo "❌ Python 3.9+ required. Current version: $current_python"
    echo ""
    echo "Since you're on Amazon Linux with Python 3.9.21, we can try to use it:"
    echo "Your Python 3.9.21 should work with the 3.9-compatible version!"
    echo ""
    
    # Let's try to use the existing Python 3.9.21
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        echo "✓ Your Python 3.9.21 is compatible!"
        python_cmd="python3"
    else
        echo "Choose installation method for newer Python:"
        echo "1) pyenv (recommended, easy to manage multiple versions)"
        echo "2) compile from source (system-wide installation)"
        echo "3) exit and use Docker instead"
        echo ""
        read -p "Enter choice (1-3): " choice
        
        case $choice in
            1)
                install_with_pyenv
                python_cmd="python"
                ;;
            2)
                install_with_compile
                python_cmd="python3"
                ;;
            3)
                echo "To use Docker instead:"
                echo "docker run -it --rm -v \$(pwd):/workspace python:3.9-slim bash"
                exit 0
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
    fi
fi

# Verify Python version
new_version=$($python_cmd --version 2>/dev/null | cut -d' ' -f2)
echo "✓ Python version: $new_version"

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "📦 Installing git..."
    sudo yum install git -y
fi

# Install the package with Python 3.9 compatible requirements
echo "📦 Installing Valkey Stress Test package (Python 3.9 compatible)..."

# Create virtual environment (recommended)
$python_cmd -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python 3.9 compatible dependencies
if [ -f "requirements_py39.txt" ]; then
    echo "Using Python 3.9 compatible requirements..."
    pip install -r requirements_py39.txt
else
    echo "Using standard requirements with manual version constraints..."
    pip install \
        'numpy>=1.20.0,<2.0.0' \
        'redis>=4.5.0,<6.0.0' \
        'psutil>=5.9.0,<6.0.0' \
        'prometheus-client>=0.19.0,<1.0.0' \
        'typer>=0.7.0,<1.0.0' \
        'pyyaml>=6.0,<7.0' \
        'h5py>=3.7.0,<4.0.0' \
        'aiofiles>=22.0,<24.0' \
        'pandas>=1.5.0,<3.0.0' \
        'rich>=12.0.0,<14.0.0'
fi

# Install the package in development mode
pip install -e .

# Verify installation
echo "🧪 Verifying installation..."
if command -v vst &> /dev/null; then
    echo "✓ vst command available"
    if vst --help > /dev/null 2>&1; then
        echo "✓ vst command works"
    else
        echo "⚠️ vst command found but may have issues"
    fi
else
    echo "❌ vst command not found"
    echo "Try: source venv/bin/activate"
fi

# Run verification script if available
if [ -f "verify_installation.py" ]; then
    echo "🔍 Running installation verification..."
    python verify_installation.py
fi

echo ""
echo "🎉 Installation complete for Python 3.9+ compatibility!"
echo ""
echo "✅ Your Python 3.9.21 should work with this version"
echo ""
echo "To use the tool:"
echo "  source venv/bin/activate  # Activate virtual environment"
echo "  vst --help                # Show available commands"
echo ""
echo "To deactivate virtual environment:"
echo "  deactivate"

# Function to install via pyenv (if needed)
install_with_pyenv() {
    echo "📦 Installing Python 3.10+ with pyenv..."
    
    # Install dependencies
    sudo yum groupinstall "Development Tools" -y
    sudo yum install git gcc openssl-devel libffi-devel bzip2-devel readline-devel sqlite-devel -y
    
    # Install pyenv
    if ! command -v pyenv &> /dev/null; then
        curl https://pyenv.run | bash
        
        # Add to bashrc
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
        
        # Load pyenv for current session
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    fi
    
    # Install Python 3.10.14
    pyenv install 3.10.14
    pyenv global 3.10.14
    
    echo "✓ Python 3.10.14 installed via pyenv"
}

# Function to install via compilation
install_with_compile() {
    echo "📦 Installing Python 3.10+ from source..."
    
    # Install dependencies
    sudo yum groupinstall "Development Tools" -y
    sudo yum install openssl-devel libffi-devel bzip2-devel readline-devel sqlite-devel -y
    
    # Download and compile Python 3.10
    cd /tmp
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
    tar xvf Python-3.10.14.tgz
    cd Python-3.10.14
    ./configure --enable-optimizations --prefix=/usr/local
    make -j $(nproc)
    sudo make altinstall
    
    # Create symlinks for easy access
    sudo ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3
    sudo ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3
    
    echo "✓ Python 3.10.14 compiled and installed"
}
