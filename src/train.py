import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import dagshub

def train_model(X_train, X_test, y_train, y_test, feature_names):
    # Initialize DagsHub (Replace with your username/repo)
    dagshub.init(repo_owner='YOUR_USERNAME', repo_name='yield-engine', mlflow=True)
    
    mlflow.set_experiment("Agriculture_Yield_XGBoost")
    
    with mlflow.start_run(run_name="Baseline_XGB"):
        # 1. Define Params
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # 2. Train
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # 3. Evaluate
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # 4. Log Feature Importance Plot
        plt.figure(figsize=(10,6))
        xgb.plot_importance(model, max_num_features=10)
        plt.title("Top 10 Agricultural Yield Drivers")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # 5. Log & Register Model
        # This makes it available via 'models:/Yield_Model/Production'
        mlflow.xgboost.log_model(
            model, 
            artifact_path="model",
            registered_model_name="Yield_Model"
        )
        
        print(f"✅ Training Complete. R2: {r2:.4f}")