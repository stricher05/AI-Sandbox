#!/bin/bash

# ML Forecasting Platform Setup Script
# This script sets up the complete environment

set -e

echo "🚀 Setting up ML Forecasting Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $python_version"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data models logs monitoring/dashboards

# Generate sample data
print_status "Generating sample data..."
python3 -c "
import sys
sys.path.append('src')
from data.preprocessing import generate_sample_data
generate_sample_data('data/sample_energy_data.csv')
print('✅ Sample data generated')
"

# Run a quick test
print_status "Running quick validation test..."
python3 -c "
import sys
sys.path.append('src')
from config import ConfigManager
config = ConfigManager('config/default_config.yaml')
config.load_config()
print('✅ Configuration validation passed')
"

# Create startup scripts
print_status "Creating startup scripts..."

# Training script
cat > run_training.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting ML training pipeline..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
export PYTHONPATH=$PWD/src:$PYTHONPATH
python src/train_model.py --config config/default_config.yaml
echo "✅ Training completed!"
EOF
chmod +x run_training.sh

# API startup script
cat > run_api.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting ML forecasting API..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
export PYTHONPATH=$PWD/src:$PYTHONPATH
python src/api/forecast_api.py
EOF
chmod +x run_api.sh

# Summary
print_status "Setup completed successfully! 🎉"
echo ""
echo "📋 Next steps:"
echo "  1. Run training: ./run_training.sh"
echo "  2. Start API: ./run_api.sh"
echo "  3. Or use Docker: docker-compose up"
echo ""
echo "🌟 Your production-ready ML forecasting platform is now ready!"
