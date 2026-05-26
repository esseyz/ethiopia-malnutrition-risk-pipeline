import pandas as pd
import numpy as np

def analyze_clinical_targets(file_path):
    """
    Cleans DHS-specific structural codes, normalizes metric scaling,
    and maps anthropometric Z-scores to clinical diagnostic classes.
    """
    print("=== Phase 2: Clinical Target Variable Analysis ===")
    df = pd.read_stata(file_path, convert_categoricals=False)
    
    # Target columns extraction
    targets = df[['hw70', 'hw72']].copy()
    targets.columns = ['stunting_raw', 'wasting_raw']
    
    # 1. Filter out DHS structural flag values (9996, 9998, 9999)
    # Any value > 9000 represents a measurement error or missing flag
    clean_mask = (targets['stunting_raw'] < 9000) & (targets['wasting_raw'] < 9000)
    cleaned_targets = targets[clean_mask].copy()
    
    # 2. Rescale from DHS integer storage back to actual clinical Z-Scores
    cleaned_targets['stunting_z'] = cleaned_targets['stunting_raw'] / 100.0
    cleaned_targets['wasting_z'] = cleaned_targets['wasting_raw'] / 100.0
    
    # 3. Apply WHO Epidemiological Categorization for Stunting
    def map_to_who_category(z_score):
        if z_score < -3.0:
            return "Severe"
        elif z_score < -2.0:
            return "Moderate"
        else:
            return "Normal"
            
    cleaned_targets['stunting_class'] = cleaned_targets['stunting_z'].apply(map_to_who_category)
    
    print(f"[OK] Filtered out structural flags. Valid sample size: {len(cleaned_targets)} children.")
    print("\n--- Real-World Cleaned Z-Score Distribution ---")
    print(cleaned_targets[['stunting_z', 'wasting_z']].describe())
    
    print("\n--- Target Class Imbalance (Stunting Category Profile) ---")
    counts = cleaned_targets['stunting_class'].value_counts()
    percentages = cleaned_targets['stunting_class'].value_counts(normalize=True) * 100
    for idx in counts.index:
        print(f" {idx:<10} Count: {counts[idx]:>4} | Proportion: {percentages[idx]:.2f}%")
        
    return cleaned_targets

if __name__ == "__main__":
    raw_data_path = "data/raw/ETKR71FL.DTA"
    analyze_clinical_targets(raw_data_path)