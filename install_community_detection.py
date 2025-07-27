#!/usr/bin/env python3
"""Install community detection package."""

import subprocess
import sys

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def main():
    """Install required packages for community detection."""
    print("Installing community detection packages...")
    
    packages = [
        "python-louvain",  # For community detection
        "networkx>=2.5",   # For graph operations
    ]
    
    success = True
    for package in packages:
        if not install_package(package):
            success = False
    
    if success:
        print("\n🎉 All packages installed successfully!")
        print("You can now use community clustering in FSRA Improved model.")
    else:
        print("\n❌ Some packages failed to install.")
        print("Community clustering will use fallback K-means clustering.")

if __name__ == "__main__":
    main()
