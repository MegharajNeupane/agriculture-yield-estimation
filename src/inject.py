import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(path: str):
    try:
        df = pd.read_csv(path)
        logging.info(f"Ingested data with shape: {df.shape}")
        
        # Basic Integrity Check
        if df.isnull().values.any():
            logging.warning("Data contains null values. Preprocessing will handle this.")
            
        return df
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        raise