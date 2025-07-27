#!/usr/bin/env python3
"""
Script to view and analyze training logs.
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def list_experiments(log_dir="logs"):
    """List all available experiments."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory {log_dir} does not exist.")
        return []
    
    experiments = []
    for exp_dir in log_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name != '.gitkeep':
            experiments.append(exp_dir.name)
    
    return sorted(experiments)


def view_experiment_logs(experiment_name, log_dir="logs"):
    """View logs for a specific experiment."""
    exp_path = Path(log_dir) / experiment_name
    
    if not exp_path.exists():
        print(f"Experiment {experiment_name} not found.")
        return
    
    print(f"📁 Experiment: {experiment_name}")
    print(f"📂 Location: {exp_path}")
    print("=" * 60)
    
    # Show available files
    files = list(exp_path.glob("*"))
    print("📄 Available files:")
    for file in files:
        size = file.stat().st_size / 1024  # KB
        print(f"   {file.name} ({size:.1f} KB)")
    print()
    
    # Show config if available
    config_file = exp_path / "config.json"
    if config_file.exists():
        print("⚙️  Configuration:")
        with open(config_file, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            print(f"   {key}: {value}")
        print()
    
    # Show recent training log entries
    training_log = exp_path / "training.log"
    if training_log.exists():
        print("📝 Recent training log entries:")
        with open(training_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Show last 20 lines
        for line in lines[-20:]:
            print(f"   {line.strip()}")
        print()
    
    # Show metrics summary
    metrics_file = exp_path / "metrics_summary.json"
    if metrics_file.exists():
        print("📊 Metrics Summary:")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if metrics:
            last_epoch = metrics[-1]
            print(f"   Total epochs: {len(metrics)}")
            print(f"   Last epoch metrics:")
            for key, value in last_epoch['metrics'].items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.6f}")
                else:
                    print(f"     {key}: {value}")


def plot_training_metrics(experiment_name, log_dir="logs"):
    """Plot training metrics for an experiment."""
    exp_path = Path(log_dir) / experiment_name
    metrics_file = exp_path / "metrics_summary.json"
    
    if not metrics_file.exists():
        print(f"No metrics file found for experiment {experiment_name}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    if not metrics_data:
        print("No metrics data available")
        return
    
    # Extract epoch-level metrics
    epoch_metrics = [entry for entry in metrics_data if 'batch' not in entry]
    
    if not epoch_metrics:
        print("No epoch-level metrics found")
        return
    
    # Create DataFrame
    df_data = []
    for entry in epoch_metrics:
        row = {'epoch': entry['epoch']}
        row.update(entry['metrics'])
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics - {experiment_name}', fontsize=16)
    
    # Loss plot
    if 'avg_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['avg_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Average Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in df.columns:
        axes[0, 1].plot(df['epoch'], df['accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Memory usage plot
    if 'gpu_memory_used' in df.columns:
        axes[1, 0].plot(df['epoch'], df['gpu_memory_used'], 'r-', linewidth=2)
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Time plot
    if 'epoch_time' in df.columns:
        axes[1, 1].plot(df['epoch'], df['epoch_time'], 'm-', linewidth=2)
        axes[1, 1].set_title('Epoch Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = exp_path / "training_metrics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 Plot saved to: {plot_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="View training logs")
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--experiment', type=str, help='View specific experiment')
    parser.add_argument('--plot', type=str, help='Plot metrics for experiment')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiments(args.log_dir)
        if experiments:
            print("📁 Available experiments:")
            for exp in experiments:
                print(f"   {exp}")
        else:
            print("No experiments found.")
    
    elif args.experiment:
        view_experiment_logs(args.experiment, args.log_dir)
    
    elif args.plot:
        try:
            plot_training_metrics(args.plot, args.log_dir)
        except ImportError:
            print("matplotlib and pandas are required for plotting.")
            print("Install with: pip install matplotlib pandas")
    
    else:
        # Default: list experiments
        experiments = list_experiments(args.log_dir)
        if experiments:
            print("📁 Available experiments:")
            for exp in experiments:
                print(f"   {exp}")
            print("\nUse --experiment <name> to view details")
            print("Use --plot <name> to plot metrics")
        else:
            print("No experiments found.")


if __name__ == "__main__":
    main()
