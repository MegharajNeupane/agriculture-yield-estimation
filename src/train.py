import mlflow
import mlflow.xgboost
import xgboost as xgb
# Import both to handle different sklearn versions
from sklearn.metrics import mean_squared_error, r2_score 
import numpy as np
import matplotlib.pyplot as plt
import dagshub

def train_model(X_train, X_test, y_train, y_test, feature_names):
    dagshub.init(repo_owner='MegharajNeupane', repo_name='agriculture-yield-estimation', mlflow=True)
    
    mlflow.set_experiment("Agri_Yield_XGBoost")
    
    with mlflow.start_run(run_name="Professional_XGB_Run"):
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "objective": "reg:squarederror",
            "random_state": 42
        }
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # --- EVALUATION FIX ---
        preds = model.predict(X_test)
        
        # Calculate RMSE manually to avoid version conflicts
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_test, preds)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        # ----------------------

        # Feature Importance Plot
        plt.figure(figsize=(12, 8))
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-15:]
        plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
        plt.title("Top Agricultural Yield Drivers")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        mlflow.xgboost.log_model(
            model, 
            artifact_path="model",
            registered_model_name="Yield_Model_XGB"
        )
        
        print(f"✅ Training Complete. R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")