#!/bin/bash

# Setup script for Python 3.9 compatibility
# Uses the existing Python 3.9.21 installation

set -e

echo "🚀 Setting up Valkey Stress Test with Python 3.9..."

# Check Python version
current_python=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "not found")
echo "Current Python version: $current_python"

# Verify we have Python 3.9+
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "✓ Python 3.9+ available"
    python_cmd="python3"
else
    echo "❌ Python 3.9+ required but not found"
    exit 1
fi

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "📦 Installing git..."
    sudo yum install git -y
fi

# Test SSL support (important for pip)
echo "🔧 Testing SSL support for pip..."
if python3 -c "import ssl; print('SSL support available')" 2>/dev/null; then
    echo "✓ SSL support working"
else
    echo "⚠️  SSL support issue detected, but continuing..."
fi

# Use Apollo Python directly with user installs (more reliable than venv on this system)
echo "📦 Setting up Python environment..."
# Use the full path to the working Apollo Python
PYTHON_CMD="/apollo/env/AmazonAwsCli/bin/python3"

# Check if pip is available
if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo "❌ pip not available in Apollo Python environment"
    exit 1
fi

# Create a function to use the correct python
python() {
    $PYTHON_CMD "$@"
}

# Create a function to use pip with user installs
pip() {
    $PYTHON_CMD -m pip --user "$@"
}

# Export functions for use in subshells
export -f python
export -f pip

# Upgrade pip (user install)
echo "📦 Upgrading pip..."
$PYTHON_CMD -m pip install --user --upgrade pip

# Install dependencies using Python 3.9 requirements
echo "📦 Installing dependencies for Python 3.9..."
$PYTHON_CMD -m pip install --user -r requirements_py39.txt

# Install the package in development mode
echo "📦 Installing valkey-stress-test package..."
$PYTHON_CMD -m pip install --user -e .

# Add user bin directory to PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
echo "🧪 Verifying installation..."
if command -v vst &> /dev/null; then
    echo "✓ vst command available"
    if vst --help > /dev/null 2>&1; then
        echo "✓ vst command works"
    else
        echo "⚠️  vst command found but may have issues"
    fi
else
    echo "❌ vst command not found in PATH"
    echo "The package was installed but the vst command may not be in PATH."
    echo "Try adding ~/.local/bin to your PATH:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Run verification script if available
if [ -f "setup_check_py39.py" ]; then
    echo "🔍 Running Python 3.9 compatibility check..."
    $PYTHON_CMD setup_check_py39.py
elif [ -f "verify_installation.py" ]; then
    echo "🔍 Running installation verification..."
    $PYTHON_CMD verify_installation.py
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To use the tool:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\"  # Add to PATH if needed"
echo "  vst --help                               # Show available commands"
echo ""
echo "To make PATH permanent, add this to your ~/.zshrc:"
echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
echo ""
echo "Note: This setup uses Python 3.9 compatible package versions."
echo "Some newer features may not be available, but core functionality should work."
