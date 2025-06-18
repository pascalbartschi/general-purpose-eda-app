@echo off
:: Script to set up Conda environment for EDA App
echo ========================================
echo Setting up Conda environment for EDA App
echo ========================================
echo.

:: Check if conda is available
call conda --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not available in PATH.
    echo Please install Miniconda or Anaconda and try again.
    exit /b 1
)

:: Create conda environment from environment.yml
echo Creating conda environment from environment.yml...
call conda env create -f environment.yml

if %ERRORLEVEL% neq 0 (
    echo Failed to create conda environment.
    echo Trying alternative method...
    
    :: Try creating environment manually
    call conda create -n eda_app python=3.9 -y
    if %ERRORLEVEL% neq 0 (
        echo Failed to create conda environment.
        exit /b 1
    )
    
    :: Activate environment
    call conda activate eda_app
    if %ERRORLEVEL% neq 0 (
        echo Failed to activate conda environment.
        exit /b 1
    )
    
    :: Install requirements
    call pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements.
        exit /b 1
    )
) else (
    :: Activate environment
    call conda activate eda_app
)

echo.
echo Environment set up successfully!
echo.

:: Check if GTK3 is installed for WeasyPrint
echo Checking for GTK3 installation (required for PDF reports)...
if exist "C:\Program Files\GTK3-Runtime\bin" (
    echo GTK3 found in Program Files
    set PATH=%PATH%;C:\Program Files\GTK3-Runtime\bin
    echo Added to PATH for this session
) else if exist "C:\Program Files (x86)\GTK3-Runtime\bin" (
    echo GTK3 found in Program Files (x86)
    set PATH=%PATH%;C:\Program Files (x86)\GTK3-Runtime\bin
    echo Added to PATH for this session
) else (
    echo GTK3 not found. PDF report generation may not work.
    echo.
    echo Would you like to check the GTK installation guide?
    choice /C YN /M "Open GTK installation guide? (Y/N)"
    if %ERRORLEVEL% equ 1 (
        start https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
        echo.
        echo After installing GTK3, run fix_gtk_windows.bat to verify your setup.
    )
)

echo.
echo ===== NEXT STEPS =====
echo 1. To activate the environment, run: conda activate eda_app
echo 2. To run the app, run: streamlit run app.py
echo 3. To fix GTK issues for PDF reports, run: fix_gtk_windows.bat
echo.
