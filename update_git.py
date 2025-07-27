#!/usr/bin/env python3
"""Quick Git update script for ViT-CNN-crossview."""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def run_command(command, check=True):
    """Run a shell command and return success status."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return False


def get_git_status():
    """Get current git status."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return ""


def main():
    """Main update function."""
    print("=" * 60)
    print("ViT-CNN-crossview Git Update Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("train.py").exists() or not Path("src").exists():
        print("Error: Please run this script from the ViT-CNN-crossview directory")
        return 1
    
    # Check if it's a git repository
    if not Path(".git").exists():
        print("This is not a Git repository. Running initial setup...")
        return subprocess.call([sys.executable, "upload_to_git.py"])
    
    print("✓ Found Git repository")
    
    # Check for changes
    status = get_git_status()
    if not status:
        print("No changes to commit.")
        return 0
    
    print(f"Found {len(status.splitlines())} changed files")
    
    # Show status
    print("\n1. Current Git status:")
    run_command("git status --short")
    
    # Add all changes
    print("\n2. Adding all changes...")
    if not run_command("git add ."):
        print("Failed to add changes")
        return 1
    
    # Get commit message
    print("\n3. Creating commit...")
    
    # Auto-generate commit message based on changes
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Check what files were changed
    changed_files = status.splitlines()
    
    if any("trainer" in line for line in changed_files):
        commit_msg = f"feat: update trainer module and components ({timestamp})"
    elif any("config" in line for line in changed_files):
        commit_msg = f"fix: update configuration and imports ({timestamp})"
    elif any("model" in line for line in changed_files):
        commit_msg = f"feat: update model architecture ({timestamp})"
    elif any("README" in line or "md" in line for line in changed_files):
        commit_msg = f"docs: update documentation ({timestamp})"
    else:
        commit_msg = f"update: general improvements and fixes ({timestamp})"
    
    print(f"Commit message: {commit_msg}")
    
    if not run_command(f'git commit -m "{commit_msg}"'):
        print("Commit failed")
        return 1
    
    # Push changes
    print("\n4. Pushing to GitHub...")
    if run_command("git push origin main"):
        print("\n🎉 SUCCESS! Changes have been pushed to GitHub!")
        print("Repository URL: https://github.com/newhouse10086/ViT-CNN-crossview")
    else:
        print("\n⚠️ Push failed. Possible reasons:")
        print("1. Authentication issues")
        print("2. Network connectivity")
        print("3. Remote repository conflicts")
        print("\nTry running: git push origin main")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
