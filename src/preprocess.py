import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_pipeline(df, target_column='Yield'):
    # 1. Drop Date columns and Year (Year can be kept if relevant, but often causes leakage)
    # We also drop the target to create our feature set X
    X = df.drop(columns=[target_column, 'Planting_Date', 'Harvest_Date', 'Year'])
    y = df[target_column]
    
    # 2. Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Create Transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Fit and Transform
    # We fit on train and transform both to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 7. Save the preprocessor (Essential for the FastAPI production API)
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # Get feature names after one-hot encoding for the XGBoost plot
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + cat_feature_names
    
    return X_train_processed, X_test_processed, y_train, y_test, feature_names