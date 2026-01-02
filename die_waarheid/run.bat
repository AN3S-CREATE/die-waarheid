@echo off
REM Die Waarheid - Windows Startup Script
REM Comprehensive startup and configuration script for Windows

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_NAME=Die Waarheid
set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%venv
set PYTHON_CMD=python

REM Colors (using ANSI escape codes)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Functions
:print_header
echo.
echo %BLUE%========================================%NC%
echo %BLUE%%~1%NC%
echo %BLUE%========================================%NC%
exit /b

:print_success
echo %GREEN%[OK] %~1%NC%
exit /b

:print_error
echo %RED%[ERROR] %~1%NC%
exit /b

:print_warning
echo %YELLOW%[WARNING] %~1%NC%
exit /b

:print_info
echo %BLUE%[INFO] %~1%NC%
exit /b

REM Check Python version
:check_python
call :print_header "Checking Python Installation"

%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is not installed or not in PATH"
    exit /b 1
)

for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
call :print_success "Python %PYTHON_VERSION% found"
exit /b 0

REM Setup virtual environment
:setup_venv
call :print_header "Setting Up Virtual Environment"

if exist "%VENV_DIR%" (
    call :print_warning "Virtual environment already exists"
    set /p RECREATE="Recreate? (y/n): "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q "%VENV_DIR%"
        %PYTHON_CMD% -m venv "%VENV_DIR%"
        call :print_success "Virtual environment created"
    )
) else (
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    call :print_success "Virtual environment created"
)

call "%VENV_DIR%\Scripts\activate.bat"
call :print_success "Virtual environment activated"
exit /b 0

REM Install dependencies
:install_dependencies
call :print_header "Installing Dependencies"

if not exist "%PROJECT_DIR%requirements.txt" (
    call :print_error "requirements.txt not found"
    exit /b 1
)

python -m pip install --upgrade pip setuptools wheel
pip install -r "%PROJECT_DIR%requirements.txt"

call :print_success "Dependencies installed"
exit /b 0

REM Validate configuration
:validate_config
call :print_header "Validating Configuration"

cd /d "%PROJECT_DIR%"

python config.py >nul 2>&1
if errorlevel 1 (
    call :print_error "Configuration validation failed"
    call :print_info "Run: python config.py"
    exit /b 1
)

call :print_success "Configuration is valid"
exit /b 0

REM Create directories
:create_directories
call :print_header "Creating Directories"

if not exist "%PROJECT_DIR%data\audio" mkdir "%PROJECT_DIR%data\audio"
if not exist "%PROJECT_DIR%data\text" mkdir "%PROJECT_DIR%data\text"
if not exist "%PROJECT_DIR%data\temp" mkdir "%PROJECT_DIR%data\temp"
if not exist "%PROJECT_DIR%data\output\mobitables" mkdir "%PROJECT_DIR%data\output\mobitables"
if not exist "%PROJECT_DIR%data\output\reports" mkdir "%PROJECT_DIR%data\output\reports"
if not exist "%PROJECT_DIR%data\output\exports" mkdir "%PROJECT_DIR%data\output\exports"
if not exist "%PROJECT_DIR%credentials" mkdir "%PROJECT_DIR%credentials"
if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"

call :print_success "Directories created"
exit /b 0

REM Check environment file
:check_env
call :print_header "Checking Environment Configuration"

if not exist "%PROJECT_DIR%.env" (
    if exist "%PROJECT_DIR%.env.example" (
        call :print_warning ".env file not found"
        call :print_info "Creating .env from .env.example"
        copy "%PROJECT_DIR%.env.example" "%PROJECT_DIR%.env" >nul
        call :print_warning "Please edit .env with your API keys"
    ) else (
        call :print_error ".env.example not found"
        exit /b 1
    )
) else (
    call :print_success ".env file found"
)
exit /b 0

REM Run tests
:run_tests
call :print_header "Running Tests"

where pytest >nul 2>&1
if errorlevel 1 (
    call :print_warning "pytest not installed, skipping tests"
    call :print_info "Install with: pip install pytest"
    exit /b 0
)

pytest "%PROJECT_DIR%tests\" -v --tb=short
call :print_success "Tests completed"
exit /b 0

REM Start application
:start_app
call :print_header "Starting %PROJECT_NAME%"

cd /d "%PROJECT_DIR%"

call :print_info "Application starting on http://localhost:8501"
call :print_info "Press Ctrl+C to stop"

streamlit run app.py --logger.level=info
exit /b 0

REM Show menu
:show_menu
echo.
echo Die Waarheid - Main Menu
echo ========================
echo 1. Full Setup (venv + dependencies + validation)
echo 2. Install Dependencies Only
echo 3. Validate Configuration
echo 4. Run Tests
echo 5. Start Application
echo 6. Create Directories
echo 7. Check Environment
echo 8. Exit
echo.
set /p choice="Select option (1-8): "
exit /b 0

REM Main execution
:main
call :print_header "%PROJECT_NAME% Startup Script"

if "%~1"=="" (
    goto interactive_mode
) else (
    goto command_mode
)

:command_mode
if /i "%~1"=="setup" (
    call :check_python
    call :setup_venv
    call :install_dependencies
    call :create_directories
    call :check_env
    call :validate_config
    call :print_success "Setup complete! Run 'run.bat start' to begin"
    exit /b 0
) else if /i "%~1"=="start" (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    call :start_app
    exit /b 0
) else if /i "%~1"=="test" (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    call :run_tests
    exit /b 0
) else if /i "%~1"=="validate" (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    call :validate_config
    exit /b 0
) else (
    call :print_error "Unknown command: %~1"
    echo Usage: %0 {setup^|start^|test^|validate}
    exit /b 1
)

:interactive_mode
:menu_loop
call :show_menu

if "%choice%"=="1" (
    call :check_python
    call :setup_venv
    call :install_dependencies
    call :create_directories
    call :check_env
    call :validate_config
    call :print_success "Setup complete!"
    goto menu_loop
) else if "%choice%"=="2" (
    call :setup_venv
    call :install_dependencies
    goto menu_loop
) else if "%choice%"=="3" (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    call :validate_config
    goto menu_loop
) else if "%choice%"=="4" (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    call :run_tests
    goto menu_loop
) else if "%choice%"=="5" (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    call :start_app
    goto menu_loop
) else if "%choice%"=="6" (
    call :create_directories
    goto menu_loop
) else if "%choice%"=="7" (
    call :check_env
    goto menu_loop
) else if "%choice%"=="8" (
    call :print_info "Exiting..."
    exit /b 0
) else (
    call :print_error "Invalid option"
    goto menu_loop
)

endlocal
