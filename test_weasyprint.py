"""
Test script for WeasyPrint and GTK installation on Windows
This script checks if WeasyPrint is correctly installed and can generate PDFs.
"""
import os
import sys

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

try:
    print("Importing weasyprint...")
    import weasyprint
    print("WeasyPrint version:", weasyprint.__version__)
    
    print("\nChecking environment variables:")
    if 'PATH' in os.environ:
        paths = os.environ['PATH'].split(';')
        print(f"Number of PATH entries: {len(paths)}")
        gtk_paths = [p for p in paths if 'gtk' in p.lower()]
        if gtk_paths:
            print("Found GTK paths in PATH:")
            for path in gtk_paths:
                print(f"  - {path}")
        else:
            print("No GTK paths found in PATH environment variable")
    
    print("\nTrying to generate a PDF...")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WeasyPrint Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2C3E50; }
            .container { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>WeasyPrint is working!</h1>
        <div class="container">
            <p>If you can see this PDF, your GTK installation is correctly set up.</p>
            <p>You should be able to generate PDF reports in the EDA app without issues.</p>
        </div>
    </body>
    </html>
    """
    
    pdf_file = "weasyprint_test.pdf"
    weasyprint.HTML(string=html_content).write_pdf(pdf_file)
    
    if os.path.exists(pdf_file):
        file_size = os.path.getsize(pdf_file)
        print(f"Success! PDF created: {pdf_file} ({file_size} bytes)")
        print(f"PDF location: {os.path.abspath(pdf_file)}")
    else:
        print("Error: PDF file was not created")
        
except ImportError as e:
    print("Error importing WeasyPrint:", str(e))
    print("\nPlease check that WeasyPrint is installed:")
    print("pip install weasyprint")
    
except Exception as e:
    print("\nError during PDF generation:", str(e))
    print("\nThis is likely a GTK-related error. Please check your GTK installation:")
    print("1. Install GTK3 from: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases")
    print("2. Make sure to check 'Add install directory to PATH' during installation")
    print("3. Restart your computer after installation")
    print("4. Run fix_gtk_windows.bat to verify your setup")

print("\nTest completed.")
