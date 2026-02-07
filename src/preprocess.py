import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_pipeline(df, target_column='Yield_kg_per_hectare'):
    # 1. Feature Selection (Removing IDs/Leaky columns)
    # Note: Adjust column names based on the specific Kaggle CSV headers
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Save Scaler (Essential for Production API)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()