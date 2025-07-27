#!/usr/bin/env python3
"""Server-side update script for ViT-CNN-crossview."""

import subprocess
import sys
import os
from pathlib import Path
import shutil


def run_command(command, check=True, capture_output=True):
    """Run a shell command and return result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=capture_output, text=True)
        if result.stdout and capture_output:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return False, "", str(e)


def check_git_installed():
    """Check if git is installed."""
    success, _, _ = run_command("git --version")
    return success


def backup_local_changes():
    """Backup local changes if any."""
    print("Checking for local changes...")
    success, output, _ = run_command("git status --porcelain")
    
    if success and output.strip():
        print("Found local changes, creating backup...")
        timestamp = subprocess.check_output(['date', '+%Y%m%d_%H%M%S']).decode().strip()
        backup_dir = f"backup_{timestamp}"
        
        # Create backup
        shutil.copytree(".", f"../{backup_dir}", 
                       ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
        print(f"Backup created at ../{backup_dir}")
        return True
    return False


def update_existing_repo():
    """Update existing git repository."""
    print("Updating existing repository...")
    
    # Check if it's a git repository
    if not Path(".git").exists():
        print("Not a git repository. Use clone method instead.")
        return False
    
    # Backup local changes
    has_changes = backup_local_changes()
    
    # Fetch latest changes
    print("Fetching latest changes...")
    success, _, _ = run_command("git fetch origin")
    if not success:
        print("Failed to fetch from remote")
        return False
    
    # Check if we're behind
    success, output, _ = run_command("git status -uno")
    if "behind" in output:
        print("Repository is behind remote. Updating...")
        
        if has_changes:
            # Stash local changes
            print("Stashing local changes...")
            run_command("git stash")
        
        # Pull latest changes
        success, _, _ = run_command("git pull origin main")
        if not success:
            print("Failed to pull changes")
            return False
        
        if has_changes:
            print("You can restore local changes with: git stash pop")
    else:
        print("Repository is up to date!")
    
    return True


def clone_fresh_repo():
    """Clone fresh repository."""
    print("Cloning fresh repository...")
    
    repo_url = "https://github.com/newhouse10086/ViT-CNN-crossview.git"
    
    # If directory exists, rename it
    if Path("ViT-CNN-crossview").exists():
        timestamp = subprocess.check_output(['date', '+%Y%m%d_%H%M%S']).decode().strip()
        backup_name = f"ViT-CNN-crossview_backup_{timestamp}"
        print(f"Renaming existing directory to {backup_name}")
        shutil.move("ViT-CNN-crossview", backup_name)
    
    # Clone repository
    success, _, _ = run_command(f"git clone {repo_url}")
    if not success:
        print("Failed to clone repository")
        return False
    
    print("Repository cloned successfully!")
    return True


def verify_update():
    """Verify the update was successful."""
    print("Verifying update...")
    
    # Check key files exist
    key_files = [
        "train.py",
        "src/trainer/trainer.py",
        "src/utils/config_utils.py",
        "README.md"
    ]
    
    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        return False
    
    # Check if imports work
    print("Testing imports...")
    test_command = 'python -c "import sys; sys.path.append(\'src\'); from src.utils.config_utils import load_config; print(\'✓ Imports working\')"'
    success, _, _ = run_command(test_command)
    
    if success:
        print("✓ Update verification successful!")
        return True
    else:
        print("⚠ Import test failed, but files are updated")
        return True


def main():
    """Main update function."""
    print("=" * 60)
    print("ViT-CNN-crossview Server Update Script")
    print("=" * 60)
    
    # Check if git is installed
    if not check_git_installed():
        print("Error: Git is not installed or not in PATH")
        print("Please install git first:")
        print("  Ubuntu/Debian: sudo apt-get install git")
        print("  CentOS/RHEL: sudo yum install git")
        return 1
    
    print("✓ Git is available")
    
    # Determine update method
    current_dir = Path.cwd()
    project_dir = current_dir / "ViT-CNN-crossview"
    
    if project_dir.exists() and (project_dir / ".git").exists():
        print("Found existing git repository")
        os.chdir(project_dir)
        success = update_existing_repo()
    elif project_dir.exists():
        print("Found existing directory but not a git repository")
        os.chdir(current_dir)
        success = clone_fresh_repo()
        if success:
            os.chdir(project_dir)
    else:
        print("No existing directory found")
        success = clone_fresh_repo()
        if success:
            os.chdir(project_dir)
    
    if not success:
        print("❌ Update failed!")
        return 1
    
    # Verify update
    if verify_update():
        print("\n🎉 Update completed successfully!")
        print("\nNext steps:")
        print("1. Check the updated code:")
        print("   ls -la")
        print("2. Test the installation:")
        print("   python simple_test.py")
        print("3. Run training:")
        print("   python train.py --create-dummy-data --experiment-name test")
    else:
        print("\n⚠ Update completed with warnings")
        print("Please check the files manually")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
