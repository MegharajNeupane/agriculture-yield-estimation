# agriculture-yield-estimation


# 🌾 Agriculture Yield Prediction Ops



A professional MLOps pipeline for predicting agricultural yield using XGBoost, MLflow, and FastAPI.

## About the dataset
This dataset contains comprehensive agricultural records aimed at predicting crop yield based on various environmental, soil, and farming parameters. The dataset is designed to assist in yield prediction and decision-making for precision agriculture. The data includes climate factors, soil characteristics, fertilizer usage, and farming practices, making it suitable for machine learning and statistical analysis to enhance crop productivity.


## Project Architecture
- **Data Ingestion**: Modular script to handle Kaggle Agriculture datasets.
- **Experiment Tracking**: Integrated with **DagsHub** & **MLflow** for hyperparameter logging and artifact storage.
- **Model Registry**: Modern Alias-based deployment using the `@champion` tag.
- **Serving Layer**: High-performance **FastAPI** with Pydantic v2 data validation and Lifespan management.
- **Containerization**: Fully Dockerized for cloud-native deployment.



