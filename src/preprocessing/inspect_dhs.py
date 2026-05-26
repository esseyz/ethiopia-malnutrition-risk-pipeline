import pandas as pd
import numpy as np

def inspect_dhs_data(file_path):
    """
    Loads the raw EDHS Stata file, extracts core clinical/socioeconomic 
    features, and evaluates missingness patterns and structural scaling.
    """
    print("=== Phase 2: Ingesting and Inspecting EDHS Data ===")
    
    # Read Stata file (preserves value labels if needed, but we'll inspect raw structural names first)
    # convert_categoricals=False keeps numbers intact so we can handle special codes (9998, 9999) manually
    df = pd.read_stata(file_path, convert_categoricals=False)
    print(f"[OK] Successfully loaded dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    
    # Define our core architectural domain mapping
    mapping = {
        'caseid': 'case_id',
        'hw1': 'child_age_months',
        'b4': 'child_sex',
        'v012': 'mother_age',
        'v106': 'mother_education',
        'v190': 'wealth_index_quintile',
        'v102': 'residence_urban_rural',
        'hw70': 'stunting_z_score', # Target 1
        'hw71': 'underweight_z_score', # Target 2
        'hw72': 'wasting_z_score' # Target 3
    }
    
    # Intersect mapping keys with actual columns to avoid missing-key errors
    available_cols = [col for col in mapping.keys() if col in df.columns]
    sub_df = df[available_cols].rename(columns=mapping)
    
    print("--- Extracted Feature Schema Audit ---")
    print(sub_df.info())
    
    # Inspect raw target values (Highlighting DHS specific missing codes like 9996, 9998)
    print("\n--- Target Variable Distribution Check (Raw Values) ---")
    print(sub_df[['stunting_z_score', 'wasting_z_score']].describe())
    
    return sub_df

if __name__ == "__main__":
    # Update this path if your extracted folder matches a different sub-directory structure
    raw_data_path = "data/raw/ETKR71FL.DTA"
    
    try:
        extracted_df = inspect_dhs_data(raw_data_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find file at {raw_data_path}. Please verify placement within data/raw/.")