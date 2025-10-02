import joblib
import os 
import mlflow
import pickle
import cloudpickle
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.model import *

def save(filename:str, obj:object):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)


def load(filename:str) -> object:
    with open(filename, 'rb') as f:
        return cloudpickle.load(f)


def load_scaler(path):
    return joblib.load(path)

def mlflow_model(stage='Staging'):
    model_name = os.environ['APP_MODEL_NAME']
    # model_name = "st125842-a3-model"
    # 1. Define the local path for caching this model stage
    cache_path = os.path.join("models", stage)
    
    # 2. Create the cache directory if it doesn't exist
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # 3. Define the full path for the specific model file
    local_model_path = os.path.join(cache_path, model_name)

    # 4. Check if the model is already cached
    if not os.path.exists(local_model_path):
        print(f"Model not found in cache. Downloading from MLflow Registry...")
        # 5a. If not cached, download the model from the MLflow Model Registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # 5b. Save the newly downloaded model to the local cache for next time
        # joblib.dump(model, local_model_path)
        save(model_name,model)
        print(f"Model saved to local cache: {local_model_path}")

    else:
        print(f"Loading model from local cache: {local_model_path}")
    print(local_model_path)
    # 6. Load the model from the local cache file
    print(local_model_path)
    model = load(local_model_path)
    
    return model

def get_scaler_path():
    return 'models\scaler.pkl'

def get_model_path():
    model_path = 'models\model.pkl'
    # poly_model_path = os.path.join(BASE_DIR, '..', 'models', 'model_polynomial.pkl')
    # model_uri = "../mlruns/0/models/m-8f5af5f6d6404e678c6762691bd8faee/artifacts"
    model_uri = 'models\model_polynomial.pkl'
    # print(os.environ['APP_MODEL_NAME'])
    return model_path,model_uri


