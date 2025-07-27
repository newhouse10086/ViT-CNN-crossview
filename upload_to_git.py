#!/usr/bin/env python3
"""Simple Git upload script for ViT-CNN-crossview."""

import subprocess
import sys
import os
from pathlib import Path


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


def main():
    """Main upload function."""
    print("=" * 60)
    print("ViT-CNN-crossview Git Upload Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("train.py").exists() or not Path("src").exists():
        print("Error: Please run this script from the ViT-CNN-crossview directory")
        return 1
    
    print("✓ Found project files")
    
    # Configure Git user
    print("\n1. Configuring Git user...")
    run_command('git config user.name "newhouse10086"', check=False)
    run_command('git config user.email "1914906669@qq.com"', check=False)
    
    # Initialize Git repository if not already done
    print("\n2. Initializing Git repository...")
    if not Path(".git").exists():
        if not run_command("git init"):
            print("Failed to initialize Git repository")
            return 1
    else:
        print("Git repository already exists")
    
    # Add all files
    print("\n3. Adding files to Git...")
    if not run_command("git add ."):
        print("Failed to add files")
        return 1
    
    # Create commit
    print("\n4. Creating commit...")
    commit_message = """Initial commit: ViT-CNN-crossview framework

- Complete refactoring of FSRA project for PyTorch 2.1
- Hybrid ViT-CNN architecture for cross-view geo-localization
- Advanced data handling with dummy dataset support
- Comprehensive metrics and visualization tools
- Professional project structure and documentation
- Ready for research and production use"""
    
    if not run_command(f'git commit -m "{commit_message}"'):
        print("Commit may have failed, but continuing...")
    
    # Set up remote repository
    print("\n5. Setting up remote repository...")
    github_repo = "https://github.com/newhouse10086/ViT-CNN-crossview.git"
    
    # Check if remote exists
    result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Adding remote origin...")
        if not run_command(f'git remote add origin {github_repo}'):
            print("Failed to add remote origin")
            return 1
    else:
        print("Remote origin already exists")
    
    # Set main branch
    print("\n6. Setting main branch...")
    run_command("git branch -M main", check=False)
    
    # Push to GitHub
    print("\n7. Pushing to GitHub...")
    print("IMPORTANT: You may need to enter your GitHub credentials:")
    print("  Username: newhouse10086")
    print("  Password: Use your GitHub Personal Access Token (not your GitHub password)")
    print("\nIf you don't have a Personal Access Token:")
    print("1. Go to GitHub Settings > Developer settings > Personal access tokens")
    print("2. Generate new token with 'repo' permissions")
    print("3. Use that token as your password")
    
    input("\nPress Enter to continue with the push...")
    
    if run_command("git push -u origin main"):
        print("\n🎉 SUCCESS! Your project has been uploaded to GitHub!")
        print(f"Repository URL: {github_repo}")
        print("\nNext steps:")
        print("1. Visit your GitHub repository to verify the upload")
        print("2. Add a description and topics to your repository")
        print("3. Consider creating a release tag")
    else:
        print("\n⚠️ Push failed. This could be due to:")
        print("1. Authentication issues - check your credentials")
        print("2. Repository doesn't exist on GitHub yet")
        print("3. Network connectivity issues")
        print("\nTo create the GitHub repository:")
        print("1. Go to https://github.com/newhouse10086")
        print("2. Click 'New repository'")
        print("3. Name: ViT-CNN-crossview")
        print("4. Make it Public")
        print("5. Don't initialize with README")
        print("6. Create repository")
        print("7. Then run this script again")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
