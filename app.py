import sys
import os
import time

# Ensure root directory is on path to clear import footprints
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing.inspect_dhs import inspect_dhs_data
from src.preprocessing.target_analysis import analyze_clinical_targets
from src.preprocessing.statistical_inference import run_statistical_inference
from src.models.export_artifacts import export_production_assets

def run_complete_pipeline():
    """
    The master orchestrator execution loop. 
    Coordinates end-to-end ingestion, inference testing, and artifact deployment.
    """
    raw_data_path = "data/raw/ETKR71FL.DTA"
    
    print("=" * 60)
    print("  ETHIOPIA MALNUTRITION RISK PREDICTIVE PIPELINE ORCHESTRATOR  ")
    print("=" * 60)
    start_time = time.time()
    
    # Fail-fast safeguard check
    if not os.path.exists(raw_data_path):
        print(f"\n[FATAL ERROR] Immutable source data missing at: '{raw_data_path}'")
        print("Please fetch the EDHS Stata Children's Recode (.DTA) file per the instructions.")
        sys.exit(1)
        
    try:
        # Phase 2: Ingestion & Structural Inspection
        print("\n[LAUNCHING PHASE 2] Auditing Raw Survey Schema...")
        inspect_dhs_data(raw_data_path)
        
        # Phase 2b: Clinical Class Evaluation
        print("\n[LAUNCHING PHASE 2b] Evaluating WHO Clinical Imbalances...")
        analyze_clinical_targets(raw_data_path)
        
        # Phase 4: Statistical Testing Guardrails
        print("\n[LAUNCHING PHASE 4] Running Chi-Square Feature Inference Verification...")
        run_statistical_inference()
        
        # Phase 5 & 6: Machine Learning Pipelines & Tableau Serialization
        print("\n[LAUNCHING PHASE 5 & 6] Initializing ColumnTransformers & Saving Artifacts...")
        export_production_assets()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"[SUCCESS] End-to-End Pipeline Completed Executing in {elapsed:.2f} seconds.")
        print(" -> Output Artifact: 'artifacts/random_forest_pipeline.pkl'")
        print(" -> UI Data Layer  : 'data/processed/tableau_malnutrition_reporting.csv'")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Pipeline execution crashed during runtime loop: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_complete_pipeline()