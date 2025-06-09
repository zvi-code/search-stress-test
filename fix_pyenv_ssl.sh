#!/bin/bash

# Quick fix script for pyenv SSL compilation issue on Amazon Linux 2
# This script addresses the specific SSL module compilation problem

set -e

echo "🔧 Fixing pyenv Python 3.10 SSL compilation issue..."

# Ensure we're in the right environment
if [[ -f ~/.bashrc ]] && grep -q "pyenv" ~/.bashrc; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# Install additional SSL-related dependencies
echo "📦 Installing additional SSL dependencies..."
sudo yum install -y openssl11-devel openssl11-libs openssl11-static

# Clean up the failed installation
echo "🧹 Cleaning up failed Python installation..."
if pyenv versions | grep -q "3.10.14"; then
    pyenv uninstall -f 3.10.14
fi

# Remove any temporary build files
sudo rm -rf /tmp/python-build.*

# Set up environment variables for SSL compilation
echo "🔧 Setting up build environment for SSL support..."
export LDFLAGS="-L/usr/lib64 -L/usr/lib64/openssl11"
export CPPFLAGS="-I/usr/include -I/usr/include/openssl11"
export PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/lib64/openssl11/pkgconfig"

# Configure Python build with explicit SSL paths
export PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-openssl=/usr --with-openssl-rpath=auto"

# Alternative approach - try with OpenSSL 1.1 paths
if [[ -d /usr/include/openssl11 ]]; then
    export PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-openssl=/usr --with-openssl-rpath=auto CPPFLAGS=-I/usr/include/openssl11 LDFLAGS=-L/usr/lib64/openssl11"
fi

echo "🐍 Installing Python 3.10.14 with proper SSL configuration..."
pyenv install 3.10.14

# Set as global version
pyenv global 3.10.14

# Verify SSL is working
echo "🧪 Testing SSL support..."
if python -c "import ssl; print(f'SSL support verified. OpenSSL version: {ssl.OPENSSL_VERSION}')"; then
    echo "✅ Success! Python 3.10.14 is now installed with working SSL support."
    echo ""
    echo "Current Python version:"
    python --version
    echo ""
    echo "You can now run the main setup script or install packages directly:"
    echo "  python -m pip install --upgrade pip"
    echo "  python -m pip install -r requirements.txt"
    echo "  python -m pip install -e ."
else
    echo "❌ SSL support still not working. Troubleshooting..."
    
    # Try to diagnose the issue
    echo ""
    echo "🔍 Diagnostic information:"
    echo "Python executable: $(which python)"
    echo "Python version: $(python --version)"
    echo "Available OpenSSL libraries:"
    find /usr -name "*ssl*.so*" 2>/dev/null | head -10
    echo ""
    echo "Python SSL module status:"
    python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path}')
try:
    import ssl
    print('SSL module imported successfully')
    print(f'OpenSSL version: {ssl.OPENSSL_VERSION}')
except ImportError as e:
    print(f'SSL import failed: {e}')
    print('Checking for _ssl module...')
    try:
        import _ssl
        print('_ssl module found')
    except ImportError:
        print('_ssl module not found - this is the core issue')
"
    
    echo ""
    echo "💡 Alternative solutions:"
    echo "1. Try the updated setup_ec2.sh script with option 2 (compile from source)"
    echo "2. Use Docker: docker run -it --rm -v \$(pwd):/workspace python:3.10-slim bash"
    echo "3. Use Python 3.9 if your code is compatible (though not recommended)"
    
    exit 1
fi
