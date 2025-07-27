#!/usr/bin/env python3
"""Quick start script for ViT-CNN-crossview."""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0


def setup_project():
    """Setup the project."""
    print("Setting up ViT-CNN-crossview project...")
    
    # Run setup script
    if Path("scripts/setup.py").exists():
        if not run_command(f"{sys.executable} scripts/setup.py"):
            print("Setup failed!")
            return False
    else:
        print("Setup script not found, creating basic directories...")
        directories = ["data", "logs", "checkpoints", "config"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    return True


def test_installation():
    """Test the installation."""
    print("Testing installation...")
    
    if Path("scripts/test_installation.py").exists():
        return run_command(f"{sys.executable} scripts/test_installation.py")
    else:
        print("Test script not found, running basic test...")
        return run_command(f"{sys.executable} test_project.py")


def run_demo():
    """Run a demo training."""
    print("Running demo training with dummy data...")
    
    # Create a simple config for demo
    demo_config = """
model:
  name: "ViTCNN"
  num_classes: 5
  num_final_clusters: 2
  use_pretrained_resnet: false
  use_pretrained_vit: false

data:
  data_dir: "demo_data"
  batch_size: 4
  num_workers: 0
  views: 2
  sample_num: 1

training:
  num_epochs: 5
  learning_rate: 0.01

system:
  gpu_ids: "0"
  use_gpu: false
  log_dir: "demo_logs"
  checkpoint_dir: "demo_checkpoints"
"""
    
    # Save demo config
    config_path = "demo_config.yaml"
    with open(config_path, 'w') as f:
        f.write(demo_config)
    
    # Run training
    success = run_command(
        f"{sys.executable} train.py "
        f"--config {config_path} "
        f"--create-dummy-data "
        f"--experiment-name demo_run"
    )
    
    # Cleanup
    os.remove(config_path)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("Check the demo_logs/ directory for training curves and metrics.")
    
    return success


def setup_git():
    """Setup git repository."""
    print("Setting up Git repository...")
    
    if Path("scripts/setup_git.py").exists():
        return run_command(f"{sys.executable} scripts/setup_git.py")
    else:
        print("Git setup script not found!")
        return False


def show_help():
    """Show help information."""
    help_text = """
ViT-CNN-crossview Quick Start Guide

Available commands:
  setup     - Setup the project (install dependencies, create directories)
  test      - Test the installation
  demo      - Run a demo training with dummy data
  git       - Setup Git repository for GitHub upload
  help      - Show this help message

Examples:
  python quick_start.py setup
  python quick_start.py test
  python quick_start.py demo
  python quick_start.py git

For manual training:
  python train.py --config config/default_config.yaml --create-dummy-data

For more information, see README.md
"""
    print(help_text)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ViT-CNN-crossview Quick Start')
    parser.add_argument('command', nargs='?', default='help',
                       choices=['setup', 'test', 'demo', 'git', 'help'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ViT-CNN-crossview Quick Start")
    print("=" * 60)
    
    if args.command == 'setup':
        success = setup_project()
        if success:
            print("\n✅ Setup completed!")
            print("Next: python quick_start.py test")
        else:
            print("\n❌ Setup failed!")
            return 1
    
    elif args.command == 'test':
        success = test_installation()
        if success:
            print("\n✅ Installation test passed!")
            print("Next: python quick_start.py demo")
        else:
            print("\n❌ Installation test failed!")
            return 1
    
    elif args.command == 'demo':
        success = run_demo()
        if success:
            print("\n✅ Demo completed!")
            print("You can now train with your own data or setup Git.")
        else:
            print("\n❌ Demo failed!")
            return 1
    
    elif args.command == 'git':
        success = setup_git()
        if success:
            print("\n✅ Git setup completed!")
        else:
            print("\n❌ Git setup failed!")
            return 1
    
    elif args.command == 'help':
        show_help()
    
    else:
        print(f"Unknown command: {args.command}")
        show_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
