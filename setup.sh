# Create the file with proper Unix line endings
cat > deploy_ai.sh << 'EOF'
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

deploy_ai() {
    clear
    echo "AI DEPLOYMENT AUTOMATION SCRIPT"
    echo "==================================="
    
    repo_url="https://github.com/Exovate/ExovateAI.git"
    repo_name=$(basename -s .git "$repo_url")
    
    print_status "Starting deployment of: $repo_name"
    
    print_status "Checking Python3 installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed. Installing..."
        sudo apt update && sudo apt install -y python3 python3-pip
    else
        print_success "Python3 is installed"
    fi
    
    print_status "Cloning repository..."
    if [ -d "$repo_name" ]; then
        print_warning "Directory exists. Pulling latest changes..."
        cd "$repo_name"
        git pull origin main
    else
        git clone "$repo_url"
        cd "$repo_name" || exit 1
    fi
    
    print_status "Installing dependencies..."
    pip3 install -r requirements.txt || pip3 install numpy requests
    
    print_status "Starting AI..."
    python3 our_ai.py
}

deploy_ai
EOF

# Make executable and run
chmod +x deploy_ai.sh
./deploy_ai.sh
