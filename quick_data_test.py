#!/usr/bin/env python3
"""Quick test for data structure."""

import os
from pathlib import Path

def main():
    data_dir = "data"
    
    print(f"Checking data structure in: {data_dir}")
    
    # Check current structure
    data_path = Path(data_dir)
    if data_path.exists():
        print("✓ Data directory exists")
        
        # Check train subdirectory
        train_path = data_path / "train"
        if train_path.exists():
            print("✓ Train directory exists")
            
            # Check satellite and drone in train
            sat_path = train_path / "satellite"
            drone_path = train_path / "drone"
            
            if sat_path.exists():
                sat_classes = len([d for d in sat_path.iterdir() if d.is_dir()])
                print(f"✓ Satellite directory exists with {sat_classes} classes")
            else:
                print("❌ Satellite directory not found in train")
            
            if drone_path.exists():
                drone_classes = len([d for d in drone_path.iterdir() if d.is_dir()])
                print(f"✓ Drone directory exists with {drone_classes} classes")
            else:
                print("❌ Drone directory not found in train")
        else:
            print("❌ Train directory not found")
    else:
        print("❌ Data directory not found")

if __name__ == "__main__":
    main()
