from src.inject import load_data
from src.preprocess import preprocess_pipeline
from src.train import train_model

def run_pipeline():
    # 1. Ingest
    DATA_PATH = "data/raw/Agri_yield_prediction.csv"
    df = load_data(DATA_PATH)
    
    # 2. Preprocess
    X_train, X_test, y_train, y_test, features = preprocess_pipeline(df)
    
    # 3. Train & Track
    train_model(X_train, X_test, y_train, y_test, features)

if __name__ == "__main__":
    run_pipeline()