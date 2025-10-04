import pytest
from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
import os
import joblib
import sys
from dotenv import load_dotenv

# print(os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import *
from models.model import LogisticRegression
# from models.model import Polynomial,LinearRegression,LogisticRegression
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

@pytest.fixture(scope="module")
def a3_model():
    user = os.environ['MLFLOW_USERNAME']
    pas = os.environ['MLFLOW_PASSWORD']
    mlflow.set_tracking_uri(f"https://{user}:{pas}@mlflow.ml.brain.cs.ait.ac.th")
    return mlflow.sklearn.load_model(model_uri="models:/st125842-a3-model/1")

def test_a3_model_load(a3_model):
    assert a3_model is not None
    # assert a3_scaler is not None
    
def test_stage_model_output_shape(a3_model):
    # model
    # mf_model = mlflow_model()
    year = 2020
    max_power =  92.15565
    engine =  1248.0
    seats =  5.0
    fuel =  0
    transmission = 1
    # scaler = load_scaler(scaler_path)
    scaler_path = get_scaler_path()
    # model_path,_ = get_model_path()

    scaler = load_scaler(scaler_path)
    # model = a3_model()
    # model_path,model_url = load_model_predict
    X = [[year, engine, max_power, seats, fuel, transmission]]
    X_scaled = scaler.transform(X)

    pred = a3_model.predict(X_scaled)
    # print(os.getprint(os.getcwd)
    # print(pred.shape)
    assert pred.shape == (1,), f"Expected prediction shape (1,), got {pred.shape}"
