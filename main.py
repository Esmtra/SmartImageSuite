#!/usr/bin/env python3
"""
SmartImageSuite - Main Application Entry Point
A comprehensive image processing and analysis toolkit.

Author: [Your Name]
Date: [Current Date]
"""

import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def main():
    """
    Main function to initialize and run the SmartImageSuite application.
    """
    print("Welcome to SmartImageSuite!")
    print("A comprehensive image processing and analysis toolkit.")
    
    # Import and initialize the GUI interface
    try:
        from gui.interface import SmartImageSuiteGUI
        app = SmartImageSuiteGUI()
        app.run()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Please ensure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 