#!/usr/bin/env python3
"""
Government Scheme Assistant — Quick Start
"""
import subprocess, sys, os

def check_and_install():
    try:
        import flask, groq, pdfplumber
        print("✓ Core packages present")
    except ImportError:
        print("Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_and_install()
    print("\n" + "="*55)
    print("  Government Scheme Assistant")
    print("="*55)
    print("  Open browser at:  http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("="*55 + "\n")
    import app
    app.init_groq()
    app.app.run(debug=False, host='0.0.0.0', port=5000)
