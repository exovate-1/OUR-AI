#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Main deployment function
deploy_ai() {
    clear
    echo "AI DEPLOYMENT AUTOMATION SCRIPT"
    echo "==================================="
    
    # Get GitHub repository URL
    repo_url="https://github.com/Exovate/ExovateAI.git"
    
    if [ -z "$repo_url" ]; then
        print_error "Repository URL is required!"
        exit 1
    fi
    
    # Extract repo name from URL
    repo_name=$(basename -s .git "$repo_url")
    
    print_status "Starting deployment of: $repo_name"
    
    # Step 1: Check if Python3 is installed
    print_status "Checking Python3 installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed. Installing..."
        sudo apt update && sudo apt install -y python3 python3-pip
    else
        print_success "Python3 is installed"
    fi
    
    # Step 2: Clone repository
    print_status "Cloning repository..."
    if [ -d "$repo_name" ]; then
        print_warning "Directory $repo_name already exists. Pulling latest changes..."
        cd "$repo_name"
        git pull origin main
    else
        git clone "$repo_url"
        cd "$repo_name" || { print_error "Failed to enter repository directory"; exit 1; }
    fi
    
    # Step 3: Check if requirements.txt exists
    print_status "Checking for requirements.txt..."
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip3 install -r requirements.txt
        
        if [ $? -eq 0 ]; then
            print_success "All dependencies installed successfully"
        else
            print_error "Failed to install some dependencies"
        fi
    else
        print_warning "requirements.txt not found. Installing basic dependencies..."
        pip3 install numpy requests
    fi
    
    # Step 4: Check if our_ai.py exists
    print_status "Looking for AI main file..."
    if [ -f "our_ai.py" ]; then
        print_success "Found our_ai.py"
        
        # Step 5: Fix any missing imports in our_ai.py
        print_status "Checking for missing imports..."
        if ! grep -q "import random" our_ai.py; then
            print_status "Adding missing 'random' import..."
            sed -i '1s/^/import random\n/' our_ai.py
        fi
        
        # Step 6: Create a startup script for easy reruns
        print_status "Creating startup script..."
        cat > start_ai.sh << 'EOF'
#!/bin/bash
echo "Starting AI..."
cd "$(dirname "$0")"
python3 our_ai.py
EOF
        chmod +x start_ai.sh
        
        # Step 7: Run the AI
        print_success "Starting AI application..."
        echo "==================================="
        python3 our_ai.py
        
    elif [ -f "app.py" ]; then
        print_success "Found app.py"
        print_success "Starting AI application..."
        echo "==================================="
        python3 app.py
    else
        print_error "No main AI file found (our_ai.py or app.py)"
        echo "Available Python files:"
        ls *.py 2>/dev/null || echo "No Python files found"
        exit 1
    fi
}

# Function to install dependencies only
install_deps() {
    print_status "Installing/updating dependencies only..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies updated"
    else
        print_error "requirements.txt not found in current directory"
    fi
}

# Function to just run the AI
run_ai() {
    print_status "Running AI..."
    
    if [ -f "our_ai.py" ]; then
        python3 our_ai.py
    elif [ -f "app.py" ]; then
        python3 app.py
    else
        print_error "No AI main file found"
    fi
}

# Function to update from git
update_ai() {
    print_status "Updating AI from repository..."
    
    if [ -d ".git" ]; then
        git pull origin main
        install_deps
        print_success "Update complete"
    else
        print_error "Not a git repository"
    fi
}

# Show usage
show_help() {
    echo "AI Deployment Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  deploy    - Full deployment (clone + install + run)"
    echo "  install   - Install dependencies only"
    echo "  run       - Run AI only"
    echo "  update    - Update from git and reinstall"
    echo "  help      - Show this help"
    echo ""
    echo "If no option provided, full deployment will run."
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy_ai
        ;;
    "install")
        install_deps
        ;;
    "run")
        run_ai
        ;;
    "update")
        update_ai
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
