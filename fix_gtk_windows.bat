@echo off
echo =========================================
echo GTK/WeasyPrint Troubleshooting for Windows
echo =========================================
echo.

echo Checking if GTK is already installed...

:: Check if GTK is in Program Files
if exist "C:\Program Files\GTK3-Runtime\bin" (
    echo Found GTK3 in Program Files.
    echo Adding to PATH temporarily for this session...
    set PATH=%PATH%;C:\Program Files\GTK3-Runtime\bin
) else if exist "C:\Program Files (x86)\GTK3-Runtime\bin" (
    echo Found GTK3 in Program Files (x86).
    echo Adding to PATH temporarily for this session...
    set PATH=%PATH%;C:\Program Files (x86)\GTK3-Runtime\bin
) else (
    echo GTK3 not found in standard locations.
    echo.
    echo Please download and install GTK3 from:
    echo https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
    echo.
    echo 1. Download the latest GTK3-Runtime .exe installer
    echo 2. Run the installer and follow the instructions
    echo 3. Make sure to check "Add install directory to PATH"
    echo 4. Restart your command prompt after installation
    echo.
    echo After installation, run this script again to verify.
    echo.
    pause
    exit /b 1
)

echo.
echo Testing WeasyPrint/GTK configuration...

:: Create a simple test script
echo import weasyprint > test_weasyprint.py
echo print("WeasyPrint version:", weasyprint.__version__) >> test_weasyprint.py
echo try: >> test_weasyprint.py
echo     weasyprint.HTML(string="<h1>Test</h1>").write_pdf("test.pdf") >> test_weasyprint.py
echo     print("Success! WeasyPrint can generate PDFs.") >> test_weasyprint.py
echo except Exception as e: >> test_weasyprint.py
echo     print("Error:", str(e)) >> test_weasyprint.py

:: Run the test
python test_weasyprint.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo There was an error running WeasyPrint.
    echo.
    echo Additional troubleshooting steps:
    echo 1. Make sure you have GTK3 installed from the link above
    echo 2. Check that the GTK bin directory is in your PATH
    echo 3. Try restarting your computer to apply PATH changes
    echo 4. If issues persist, try the manual PATH setting:
    echo    set PATH=%%PATH%%;C:\Program Files\GTK3-Runtime\bin
    echo.
    echo For more detailed help, see README.md troubleshooting section.
) else (
    echo.
    echo WeasyPrint is working correctly!
    echo You should now be able to generate PDF reports in the EDA app.
    if exist "test.pdf" (
        echo A test PDF was created successfully.
    )
)

:: Clean up test files
if exist "test_weasyprint.py" del test_weasyprint.py
if exist "test.pdf" del test.pdf

echo.
echo Troubleshooting complete.
pause
