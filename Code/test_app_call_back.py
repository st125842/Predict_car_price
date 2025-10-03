import pytest
from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
import os
import joblib
import sys

# print(os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import *
# from models.model import Polynomial,LinearRegression,LogisticRegression

def test_model_output_shape():
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
    model_path,_ = get_model_path()

    scaler = load_scaler(scaler_path)
    model = joblib.load(model_path)
    # model_path,model_url = load_model_predict
    X = [[year, engine, max_power, seats, fuel, transmission]]
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    # print(os.getprint(os.getcwd)
    # print(pred.shape)
    assert pred.shape == (1,), f"Expected prediction shape (1,), got {pred.shape}"