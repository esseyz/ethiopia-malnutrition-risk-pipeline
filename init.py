import os
import sys
import pandas as pd
import numpy as np

def initialize_workspace():
    """
    Initializes a production-grade directory structure for the 
    Child Malnutrition & Stunting Risk Assessment pipeline.
    """
    dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/preprocessing",
        "src/models",
        "artifacts",
        "dashboard"
    ]
    
    print("=== Initializing Malnutrition Analytics Workspace ===")
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created directory: {directory}")
        
    print(f"\n[INFO] Python Version: {sys.version}")
    print(f"[INFO] Pandas Version: {pd.__version__}")

if __name__ == "__main__":
    initialize_workspace()