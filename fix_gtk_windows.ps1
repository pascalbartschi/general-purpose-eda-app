# PowerShell script to fix GTK/WeasyPrint issues on Windows
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GTK/WeasyPrint Troubleshooter for Windows" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking if GTK is already installed..." -ForegroundColor Yellow

# Check GTK installation in common locations
$gtkPath = $null
if (Test-Path "C:\Program Files\GTK3-Runtime\bin") {
    $gtkPath = "C:\Program Files\GTK3-Runtime\bin"
    Write-Host "Found GTK3 in Program Files." -ForegroundColor Green
} elseif (Test-Path "C:\Program Files (x86)\GTK3-Runtime\bin") {
    $gtkPath = "C:\Program Files (x86)\GTK3-Runtime\bin"
    Write-Host "Found GTK3 in Program Files (x86)." -ForegroundColor Green
} else {
    Write-Host "GTK3 not found in standard locations." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download and install GTK3 from:" -ForegroundColor Yellow
    Write-Host "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Download the latest GTK3-Runtime .exe installer" 
    Write-Host "2. Run the installer and follow the instructions"
    Write-Host "3. Make sure to check 'Add install directory to PATH'" -ForegroundColor Yellow
    Write-Host "4. Restart your computer after installation"
    
    $install = Read-Host "Would you like to open the download page now? (y/n)"
    if ($install -eq "y") {
        Start-Process "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases"
    }
    
    Write-Host ""
    Write-Host "After installation, run this script again to verify." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit
}

# Add GTK to PATH for current session
Write-Host "Adding GTK to PATH temporarily for this session..." -ForegroundColor Yellow
$env:PATH = "$env:PATH;$gtkPath"

Write-Host ""
Write-Host "Testing WeasyPrint/GTK configuration..." -ForegroundColor Yellow

# Create test script
$testScript = @"
import sys
print('Python version:', sys.version)
try:
    import weasyprint
    print('WeasyPrint version:', weasyprint.__version__)
    html_content = '<h1>Test</h1>'
    pdf_file = 'test.pdf'
    weasyprint.HTML(string=html_content).write_pdf(pdf_file)
    print('Success! WeasyPrint can generate PDFs.')
except Exception as e:
    print('Error:', str(e))
"@

Set-Content -Path "test_weasyprint.py" -Value $testScript

# Run the test
try {
    $output = & python test_weasyprint.py
    $output | ForEach-Object { Write-Host $_ }
    
    if (Test-Path "test.pdf") {
        Write-Host ""
        Write-Host "SUCCESS: WeasyPrint is working correctly!" -ForegroundColor Green
        Write-Host "You should now be able to generate PDF reports in the EDA app." -ForegroundColor Green
        Write-Host "A test PDF was created successfully." -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "ERROR: WeasyPrint could not create a PDF." -ForegroundColor Red
    }
} catch {
    Write-Host ""
    Write-Host "There was an error running the test." -ForegroundColor Red
    Write-Host $_.Exception.Message
}

# Check if PATH contains GTK permanently
$pathPermanent = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($pathPermanent -like "*GTK*") {
    Write-Host ""
    Write-Host "GTK is in your system PATH permanently. Good!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "GTK is NOT in your system PATH permanently." -ForegroundColor Yellow
    $addPath = Read-Host "Would you like to add GTK to your system PATH permanently? (y/n)"
    if ($addPath -eq "y") {
        try {
            [Environment]::SetEnvironmentVariable("PATH", "$pathPermanent;$gtkPath", "Machine")
            Write-Host "GTK added to system PATH successfully!" -ForegroundColor Green
            Write-Host "Please restart your computer for changes to take effect." -ForegroundColor Yellow
        } catch {
            Write-Host "Error adding to PATH. You may need to run this script as Administrator." -ForegroundColor Red
        }
    }
}

# Clean up
if (Test-Path "test_weasyprint.py") { Remove-Item "test_weasyprint.py" }
if (Test-Path "test.pdf") { Remove-Item "test.pdf" }

Write-Host ""
Write-Host "Troubleshooting complete." -ForegroundColor Cyan
Read-Host "Press Enter to exit"
