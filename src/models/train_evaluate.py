import pandas as pd
import numpy as np
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure the root directory is in the path to import our partition engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.build_pipeline import load_and_split_data

def train_and_evaluate_production():
    print("=== Phase 5: Production-Ready Preprocessing & Modeling ===")
    
    # 1. Ingest clean, raw structural splits
    raw_data_path = "data/raw/ETKR71FL.DTA"
    X_train, X_test, y_train, y_test = load_and_split_data(raw_data_path)
    
    # Apply our verified engineered feature to both partitions independently
    for df in [X_train, X_test]:
        df['rural_poverty_index'] = np.where(
            (df['residence_urban_rural'] == 2) & (df['wealth_index_quintile'] <= 2), 
            1, 0
        )
    
    # 2. Define Feature Types for the Preprocessing Pipeline
    numerical_features = ['child_age_months', 'mother_age']
    categorical_features = ['child_sex', 'mother_education', 'wealth_index_quintile', 'residence_urban_rural', 'rural_poverty_index']
    
    # 3. Construct the Column Transformer Object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 4. Bind Preprocessing and Classifier into an immutable End-to-End Pipeline
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(multi_class='multinomial', class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42))
    ])
    
    # 5. Train and Evaluate
    print("\n--- Training Pipeline A: Encapsulated Logistic Regression ---")
    log_reg_pipeline.fit(X_train, y_train)
    y_pred_log = log_reg_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_log, target_names=['Normal', 'Moderate', 'Severe']))
    
    print("\n--- Training Pipeline B: Encapsulated Random Forest ---")
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Moderate', 'Severe']))
    
    return log_reg_pipeline, rf_pipeline

if __name__ == "__main__":
    train_and_evaluate_production()