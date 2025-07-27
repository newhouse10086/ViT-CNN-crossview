#!/usr/bin/env python3
"""Git setup script for ViT-CNN-crossview project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(
        command, 
        shell=True, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    if capture_output:
        return result.stdout.strip()
    return result.returncode == 0


def check_git_installed():
    """Check if git is installed."""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Git version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("Error: Git is not installed or not in PATH")
    return False


def setup_git_config():
    """Setup git configuration."""
    print("Setting up Git configuration...")
    
    # Set user name and email
    run_command('git config user.name "newhouse10086"')
    run_command('git config user.email "1914906669@qq.com"')
    
    # Set default branch name
    run_command('git config init.defaultBranch main')
    
    # Set line ending handling
    if os.name == 'nt':  # Windows
        run_command('git config core.autocrlf true')
    else:  # Linux/Mac
        run_command('git config core.autocrlf input')
    
    print("Git configuration completed")


def initialize_repository():
    """Initialize git repository."""
    print("Initializing Git repository...")
    
    # Check if already a git repository
    if Path('.git').exists():
        print("Repository already initialized")
        return True
    
    # Initialize repository
    if not run_command('git init'):
        return False
    
    # Add all files
    if not run_command('git add .'):
        return False
    
    # Initial commit
    commit_message = "Initial commit: ViT-CNN-crossview framework"
    if not run_command(f'git commit -m "{commit_message}"'):
        return False
    
    print("Repository initialized successfully")
    return True


def setup_remote_repository():
    """Setup remote repository."""
    print("Setting up remote repository...")
    
    github_repo = "https://github.com/newhouse10086/ViT-CNN-crossview.git"
    
    # Check if remote already exists
    result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        current_remote = result.stdout.strip()
        print(f"Remote origin already exists: {current_remote}")
        
        if current_remote != github_repo:
            print("Updating remote URL...")
            run_command(f'git remote set-url origin {github_repo}')
    else:
        print("Adding remote origin...")
        run_command(f'git remote add origin {github_repo}')
    
    print("Remote repository setup completed")


def create_github_repository():
    """Instructions for creating GitHub repository."""
    print("\n" + "="*60)
    print("GitHub Repository Setup Instructions")
    print("="*60)
    print("1. Go to https://github.com/newhouse10086")
    print("2. Click 'New repository'")
    print("3. Repository name: ViT-CNN-crossview")
    print("4. Description: Advanced Deep Learning Framework for UAV Geo-Localization")
    print("5. Set as Public repository")
    print("6. Do NOT initialize with README, .gitignore, or license")
    print("7. Click 'Create repository'")
    print("8. The repository URL will be: https://github.com/newhouse10086/ViT-CNN-crossview.git")
    print("="*60)


def push_to_github():
    """Push code to GitHub."""
    print("Pushing code to GitHub...")
    
    # Set upstream and push
    if not run_command('git branch -M main'):
        return False
    
    if not run_command('git push -u origin main'):
        print("Push failed. This might be because:")
        print("1. The GitHub repository doesn't exist yet")
        print("2. Authentication is required")
        print("3. Network connectivity issues")
        return False
    
    print("Code pushed to GitHub successfully!")
    return True


def setup_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
data/
!data/readme.txt
checkpoints/
logs/
*.pth
*.pt
*.ckpt
*.mat
*.xlsx
*.png
*.jpg
*.jpeg
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.tmp
*.temp
*~

# Model weights and outputs
pretrained/
outputs/
results/
wandb/
tensorboard_logs/

# Test files
test_data/
test_logs/
test_checkpoints/
test_*.yaml
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Created .gitignore file")


def create_license():
    """Create MIT license file."""
    license_content = """MIT License

Copyright (c) 2024 newhouse10086

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('LICENSE', 'w') as f:
        f.write(license_content)
    
    print("Created LICENSE file")


def main():
    """Main setup function."""
    print("=" * 60)
    print("ViT-CNN-crossview Git Setup Script")
    print("=" * 60)
    
    # Check if git is installed
    if not check_git_installed():
        print("Please install Git first: https://git-scm.com/downloads")
        return 1
    
    # Setup git configuration
    setup_git_config()
    
    # Create .gitignore and LICENSE
    setup_gitignore()
    create_license()
    
    # Initialize repository
    if not initialize_repository():
        print("Failed to initialize repository")
        return 1
    
    # Setup remote repository
    setup_remote_repository()
    
    # Show GitHub repository creation instructions
    create_github_repository()
    
    # Ask user if they want to push now
    response = input("\nHave you created the GitHub repository? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        if push_to_github():
            print("\n🎉 Repository setup completed successfully!")
            print("Your code is now available at: https://github.com/newhouse10086/ViT-CNN-crossview")
        else:
            print("\n⚠️  Repository setup completed, but push failed.")
            print("You can push manually later with: git push -u origin main")
    else:
        print("\n✅ Repository setup completed!")
        print("To push to GitHub later, run: git push -u origin main")
    
    print("\nNext steps:")
    print("1. Create the GitHub repository if you haven't already")
    print("2. Push your code: git push -u origin main")
    print("3. Add a description and topics to your GitHub repository")
    print("4. Consider adding a CONTRIBUTING.md file")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
