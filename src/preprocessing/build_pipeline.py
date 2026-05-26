import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    """
    Ingests raw data, isolates features from clinical targets, drops out-of-range 
    measurement flags, and performs a stratified train-test split to prevent leakage.
    """
    print("=== Phase 3: Executing Enterprise Preprocessing Pipeline ===")
    df = pd.read_stata(file_path, convert_categoricals=False)
    
    # 1. Define our input domain features vs. targets
    feature_mapping = {
        'hw1': 'child_age_months',
        'b4': 'child_sex',
        'v012': 'mother_age',
        'v106': 'mother_education',
        'v190': 'wealth_index_quintile',
        'v102': 'residence_urban_rural'
    }
    
    # Isolate targets raw columns
    df_cleaned = df.copy()
    
    # Drop rows where target tracking measurements are flagged as errors (>9000)
    df_cleaned = df_cleaned[(df_cleaned['hw70'] < 9000) & (df_cleaned['hw72'] < 9000)]
    
    # Process targets
    df_cleaned['stunting_z'] = df_cleaned['hw70'] / 100.0
    
    def to_clinical_class(z):
        if z < -3.0: return 2  # Severe
        elif z < -2.0: return 1 # Moderate
        else: return 0          # Normal
        
    df_cleaned['stunting_class'] = df_cleaned['stunting_z'].apply(to_clinical_class)
    
    # Extract structural subsets
    X = df_cleaned[list(feature_mapping.keys())].rename(columns=feature_mapping)
    y = df_cleaned['stunting_class']
    
    # 2. Strict Stratified Split to maintain clinical profile ratios across sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )
    
    print(f"[OK] Leakage Safeguard Active. Stratified Split Completed.")
    print(f"     -> Training Set Size: {X_train.shape[0]} records")
    print(f"     -> Validation/Test Size: {X_test.shape[0]} records")
    
    # Quick inspection of missingness inside the training features
    print("\n--- Training Set Base Missingness Profile ---")
    print(X_train.isnull().sum())
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    raw_data_path = "data/raw/ETKR71FL.DTA"
    X_train, X_test, y_train, y_test = load_and_split_data(raw_data_path)