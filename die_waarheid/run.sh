#!/bin/bash

# Die Waarheid - Startup Script
# Comprehensive startup and configuration script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Die Waarheid"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_VERSION="3.9"

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check Python version
check_python() {
    print_header "Checking Python Installation"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists"
        read -p "Recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
            print_success "Virtual environment created"
        fi
    else
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    pip install --upgrade pip setuptools wheel
    pip install -r "$PROJECT_DIR/requirements.txt"
    
    print_success "Dependencies installed"
}

# Validate configuration
validate_config() {
    print_header "Validating Configuration"
    
    cd "$PROJECT_DIR"
    
    if python3 config.py > /dev/null 2>&1; then
        print_success "Configuration is valid"
    else
        print_error "Configuration validation failed"
        print_info "Run: python3 config.py"
        exit 1
    fi
}

# Create directories
create_directories() {
    print_header "Creating Directories"
    
    mkdir -p "$PROJECT_DIR/data/audio"
    mkdir -p "$PROJECT_DIR/data/text"
    mkdir -p "$PROJECT_DIR/data/temp"
    mkdir -p "$PROJECT_DIR/data/output/mobitables"
    mkdir -p "$PROJECT_DIR/data/output/reports"
    mkdir -p "$PROJECT_DIR/data/output/exports"
    mkdir -p "$PROJECT_DIR/credentials"
    mkdir -p "$PROJECT_DIR/logs"
    
    print_success "Directories created"
}

# Check environment file
check_env() {
    print_header "Checking Environment Configuration"
    
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        if [ -f "$PROJECT_DIR/.env.example" ]; then
            print_warning ".env file not found"
            print_info "Creating .env from .env.example"
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            print_warning "Please edit .env with your API keys"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    if command -v pytest &> /dev/null; then
        pytest "$PROJECT_DIR/tests/" -v --tb=short
        print_success "Tests completed"
    else
        print_warning "pytest not installed, skipping tests"
        print_info "Install with: pip install pytest"
    fi
}

# Start application
start_app() {
    print_header "Starting $PROJECT_NAME"
    
    cd "$PROJECT_DIR"
    
    print_info "Application starting on http://localhost:8501"
    print_info "Press Ctrl+C to stop"
    
    streamlit run app.py --logger.level=info
}

# Main menu
show_menu() {
    echo ""
    echo "Die Waarheid - Main Menu"
    echo "========================"
    echo "1. Full Setup (venv + dependencies + validation)"
    echo "2. Install Dependencies Only"
    echo "3. Validate Configuration"
    echo "4. Run Tests"
    echo "5. Start Application"
    echo "6. Create Directories"
    echo "7. Check Environment"
    echo "8. Exit"
    echo ""
    read -p "Select option (1-8): " choice
}

# Main execution
main() {
    print_header "$PROJECT_NAME Startup Script"
    
    # Check if running with arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            setup)
                check_python
                setup_venv
                install_dependencies
                create_directories
                check_env
                validate_config
                print_success "Setup complete! Run './run.sh start' to begin"
                ;;
            start)
                source "$VENV_DIR/bin/activate" 2>/dev/null || true
                start_app
                ;;
            test)
                source "$VENV_DIR/bin/activate" 2>/dev/null || true
                run_tests
                ;;
            validate)
                source "$VENV_DIR/bin/activate" 2>/dev/null || true
                validate_config
                ;;
            *)
                print_error "Unknown command: $1"
                echo "Usage: $0 {setup|start|test|validate}"
                exit 1
                ;;
        esac
    else
        # Interactive mode
        while true; do
            show_menu
            
            case $choice in
                1)
                    check_python
                    setup_venv
                    install_dependencies
                    create_directories
                    check_env
                    validate_config
                    print_success "Setup complete!"
                    ;;
                2)
                    setup_venv
                    install_dependencies
                    ;;
                3)
                    source "$VENV_DIR/bin/activate" 2>/dev/null || true
                    validate_config
                    ;;
                4)
                    source "$VENV_DIR/bin/activate" 2>/dev/null || true
                    run_tests
                    ;;
                5)
                    source "$VENV_DIR/bin/activate" 2>/dev/null || true
                    start_app
                    ;;
                6)
                    create_directories
                    ;;
                7)
                    check_env
                    ;;
                8)
                    print_info "Exiting..."
                    exit 0
                    ;;
                *)
                    print_error "Invalid option"
                    ;;
            esac
        done
    fi
}

# Run main
main "$@"
