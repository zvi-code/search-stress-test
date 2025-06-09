#!/bin/bash

# Amazon Linux EC2 Setup Script for Valkey Stress Test
# This script installs Python 3.10+ and the valkey_stress_test package

set -e

echo "🚀 Setting up Valkey Stress Test on Amazon Linux EC2..."

# Check if we're on Amazon Linux
if ! grep -q "Amazon Linux" /etc/os-release 2>/dev/null; then
    echo "⚠️  This script is designed for Amazon Linux. Use INSTALL.md for other systems."
    exit 1
fi

# Check current Python version
current_python=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "not found")
echo "Current Python version: $current_python"

# Function to install via pyenv (recommended)
install_with_pyenv() {
    echo "📦 Installing Python 3.10+ with pyenv..."
    
    # Install dependencies - more comprehensive for Amazon Linux 2
    sudo yum groupinstall "Development Tools" -y
    sudo yum install -y git gcc openssl-devel libffi-devel bzip2-devel readline-devel sqlite-devel \
        zlib-devel ncurses-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel expat-devel
    
    # Install pyenv
    if ! command -v pyenv &> /dev/null; then
        curl https://pyenv.run | bash
        
        # Add to bashrc and zshrc (support both shells)
        for rcfile in ~/.bashrc ~/.zshrc; do
            if [[ -f "$rcfile" ]]; then
                if ! grep -q "pyenv" "$rcfile"; then
                    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> "$rcfile"
                    echo 'eval "$(pyenv init -)"' >> "$rcfile"
                    echo 'eval "$(pyenv virtualenv-init -)"' >> "$rcfile"
                fi
            fi
        done
        
        # Load pyenv for current session
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    fi
    
    # Set environment variables for SSL compilation
    export LDFLAGS="-L/usr/lib64/openssl -L/usr/lib64"
    export CPPFLAGS="-I/usr/include/openssl"
    export PKG_CONFIG_PATH="/usr/lib64/pkgconfig"
    
    # Configure Python build with SSL support
    export PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-openssl=/usr"
    
    echo "🔧 Installing Python 3.10.14 with SSL support..."
    pyenv install 3.10.14
    pyenv global 3.10.14
    
    # Verify SSL is working
    if ~/.pyenv/versions/3.10.14/bin/python -c "import ssl; print('SSL support verified')" 2>/dev/null; then
        echo "✓ Python 3.10.14 installed via pyenv with SSL support"
    else
        echo "❌ Python installed but SSL module not working. Trying alternative approach..."
        # Try installing with different OpenSSL paths
        export PYTHON_CONFIGURE_OPTS="--enable-shared --with-openssl=/usr --with-openssl-rpath=auto"
        pyenv uninstall -f 3.10.14
        pyenv install 3.10.14
        pyenv global 3.10.14
        
        if ~/.pyenv/versions/3.10.14/bin/python -c "import ssl; print('SSL support verified')" 2>/dev/null; then
            echo "✓ Python 3.10.14 installed via pyenv with SSL support (second attempt)"
        else
            echo "❌ SSL compilation still failing. Consider using Docker or system Python approach."
            return 1
        fi
    fi
}

# Function to install via compilation
install_with_compile() {
    echo "📦 Installing Python 3.10+ from source..."
    
    # Install dependencies - comprehensive for Amazon Linux 2
    sudo yum groupinstall "Development Tools" -y
    sudo yum install -y openssl-devel libffi-devel bzip2-devel readline-devel sqlite-devel \
        zlib-devel ncurses-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel expat-devel
    
    # Download and compile Python 3.10
    cd /tmp
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
    tar xvf Python-3.10.14.tgz
    cd Python-3.10.14
    
    # Configure with SSL support
    export LDFLAGS="-L/usr/lib64/openssl -L/usr/lib64"
    export CPPFLAGS="-I/usr/include/openssl"
    ./configure --enable-optimizations --prefix=/usr/local --with-openssl=/usr --enable-shared
    
    make -j $(nproc)
    sudo make altinstall
    
    # Create symlinks for easy access
    sudo ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3
    sudo ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3
    
    # Update shared library cache
    sudo ldconfig
    
    # Verify SSL is working
    if /usr/local/bin/python3.10 -c "import ssl; print('SSL support verified')" 2>/dev/null; then
        echo "✓ Python 3.10.14 compiled and installed with SSL support"
    else
        echo "❌ Python installed but SSL module not working"
        return 1
    fi
}

# Check if Python 3.10+ is already available
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "✓ Python 3.10+ already available"
    python_cmd="python3"
else
    echo "❌ Python 3.10+ required. Current version: $current_python"
    echo ""
    echo "Choose installation method:"
    echo "1) pyenv (recommended, easy to manage multiple versions)"
    echo "2) compile from source (system-wide installation)"
    echo "3) try Amazon Linux Extras (if available)"
    echo "4) exit and use Docker instead"
    echo ""
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            if install_with_pyenv; then
                python_cmd="python"
            else
                echo "❌ pyenv installation failed. Try option 2 or 4."
                exit 1
            fi
            ;;
        2)
            if install_with_compile; then
                python_cmd="python3"
            else
                echo "❌ Source compilation failed. Try option 4 (Docker)."
                exit 1
            fi
            ;;
        3)
            echo "📦 Trying Amazon Linux Extras..."
            # Check if amazon-linux-extras is available
            if command -v amazon-linux-extras &> /dev/null; then
                # Install Python 3.8 first, then use pip to get newer versions
                sudo amazon-linux-extras install python3.8 -y
                sudo yum install python38-devel -y
                
                # Use python3.8 to install a newer Python
                echo "⚠️  Amazon Linux Extras only provides Python 3.8. Consider using pyenv or Docker for 3.10+."
                echo "Falling back to pyenv installation..."
                if install_with_pyenv; then
                    python_cmd="python"
                else
                    echo "❌ Fallback installation failed. Use Docker (option 4)."
                    exit 1
                fi
            else
                echo "❌ Amazon Linux Extras not available. Falling back to pyenv..."
                if install_with_pyenv; then
                    python_cmd="python"
                else
                    echo "❌ Fallback installation failed. Use Docker (option 4)."
                    exit 1
                fi
            fi
            ;;
        4)
            echo "To use Docker instead:"
            echo "docker run -it --rm -v \$(pwd):/workspace python:3.10-slim bash"
            echo "# Inside container: cd /workspace && pip install -r requirements.txt && pip install -e ."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Verify Python version
new_version=$($python_cmd --version 2>/dev/null | cut -d' ' -f2)
echo "✓ Python version: $new_version"

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "📦 Installing git..."
    sudo yum install git -y
fi

# Install the package
echo "📦 Installing Valkey Stress Test package..."

# Create virtual environment (recommended)
$python_cmd -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Verify installation
echo "🧪 Verifying installation..."
if command -v vst &> /dev/null; then
    echo "✓ vst command available"
    vst --help > /dev/null && echo "✓ vst command works"
else
    echo "❌ vst command not found"
    echo "Try: source venv/bin/activate"
fi

# Run verification script
if [ -f "verify_installation.py" ]; then
    echo "🔍 Running installation verification..."
    python verify_installation.py
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To use the tool:"
echo "  source venv/bin/activate  # Activate virtual environment"
echo "  vst --help                # Show available commands"
echo ""
echo "To deactivate virtual environment:"
echo "  deactivate"
