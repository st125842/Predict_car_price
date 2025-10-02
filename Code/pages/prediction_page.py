import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import os
from xgboost import XGBRegressor
import sys
import mlflow
import mlflow.pyfunc
from models.model import *
from utils import *
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash.register_page(__name__, path='/model1')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)



text = html.Div([
    html.H1("Car Price Prediction", className="text-center my-4"),
    html.Hr()
])

# Card for model selection
model_selector_card = dbc.Card(
    dbc.CardBody([
        html.H5("Select a Model", className="card-title"),
        dcc.Dropdown(
            id="model_choice",
            options=[
                {"label": "XGBoost", "value": "xgb"},
                {"label": "Polynomial Regression", "value": "pr"},
                # {"label": "Logistic Regression", "value": "lc"},
            ],
            value="xgb",
            clearable=False,
        ),
        dbc.FormText("Choose a model to make predictions.", color="secondary", className="mt-2"),
    ]),
    className="mb-4"
)

# Creating FORM
form_section = html.Div([
    html.H2("Enter Car Specifications", className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Year", html_for="year"),
            dbc.Input(id="year", type="number", placeholder="e.g., 2020"),
            dbc.FormText("The manufacturing year of the car.", color="secondary"),
        ], md=6),
        dbc.Col([
            dbc.Label("Engine (CC)", html_for="engine"),
            dbc.Input(id="engine", type="number", placeholder="e.g., 1200"),
            dbc.FormText("Engine displacement in cubic centimeters.", color="secondary"),
        ], md=6)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Max Power (bhp)", html_for="max_power"),
            dbc.Input(id="max_power", type="number", placeholder="e.g., 88"),
            dbc.FormText("Maximum power output in bhp.", color="secondary"),
        ], md=6),
        dbc.Col([
            dbc.Label("Seats", html_for="seats"),
            dbc.Input(id="seats", type="number", placeholder="e.g., 5"),
            dbc.FormText("Number of seats in the car.", color="secondary"),
        ], md=6)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Fuel Type", html_for="fuel"),
            dcc.Dropdown(
                id="fuel",
                options=[
                    {"label": "Diesel", "value": 0},
                    {"label": "Petrol", "value": 1},
                ],
                value=0,
                clearable=False
            ),
            dbc.FormText("Select the car's fuel type.", color="secondary"),
        ], md=6),
        dbc.Col([
            dbc.Label("Transmission", html_for="transmission"),
            dcc.Dropdown(
                id="transmission",
                options=[
                    {"label": "Automatic", "value": 0},
                    {"label": "Manual", "value": 1},
                ],
                value=1,
                clearable=False
            ),
            dbc.FormText("Select the car's transmission type.", color="secondary"),
        ], md=6)
    ], className="mb-3")
])

# Submit section with a clearer layout
submit_section = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dbc.Button("Predict Price", id="submit_hardcode", color="primary", className="me-2")),
            dbc.Col(
                html.Div([
                    dbc.Label("Predicted Price:", className="me-2 lead"),
                    html.Output(id="y_hardcode", children="", className="fw-bold text-success"),
                ], className="d-flex align-items-center justify-content-end")
            )
        ])
    ]),
    className="my-4"
)

# Putting it all together in a container
layout = dbc.Container([
    text,
    model_selector_card,
    form_section,
    submit_section
], fluid=False, className="my-5")

@callback(
    Output(component_id="y_hardcode", component_property="children"),
    State(component_id="model_choice", component_property="value"),
    State(component_id="year", component_property="value"),
    State(component_id="engine", component_property="value"),
    State(component_id="max_power", component_property="value"),
    State(component_id="seats", component_property="value"),
    State(component_id="fuel", component_property="value"),
    State(component_id="transmission", component_property="value"),
    Input(component_id="submit_hardcode", component_property='n_clicks'),
    prevent_initial_call=True
)
def result(model_choice, year, engine, max_power, seats, fuel, transmission, n_clicks):
    # Set default values if inputs are None
    year = year if year is not None else 2020
    max_power = max_power if max_power is not None else 92.15565
    engine = engine if engine is not None else 1248.0
    seats = seats if seats is not None else 5.0
    fuel = fuel if fuel is not None else 0
    transmission = transmission if transmission is not None else 1

    # try:
    # Construct absolute paths to the model files
    model_path,model_url = get_model_path()
    scaler_path = get_scaler_path()
    temp = mlflow_model()
    # print(os.getcwd())
    scaler = load_scaler(os.path.join('..',str(scaler_path) ))
    X = [[year, engine, max_power, seats, fuel, transmission]]
    X_scaled = scaler.transform(X)
    # print(X_scaled)
    # print(model_url, model_path)
    # print(scaler_path)
    # print(os.path.join('..','models', 'model.pkl'))
    # print('-------------------')
    # temp = 
    
    # print(model)
    if model_choice == 'xgb':
        model = joblib.load(os.path.join('..',str(model_path) ))
        pred = model.predict(X_scaled)
        # pred = load_model_predict(model_choice,model_path,X_scaled)
    elif model_choice == 'pr':
    
        model = joblib.load(os.path.join('..',str(model_url) ))
        # This predict call assumes your custom Polynomial class has a predict method that takes a boolean
        pred = model.predict(X_scaled, True)

    else:
        return "Please select a valid model."
    print(pred)
    final_price = np.exp(pred)
    return f"{final_price[0]:,.2f} Bath"

    # except FileNotFoundError as e:
    #     return f"Error: One or more model files were not found. Please ensure the files exist at the correct path: {e}"
    # except Exception as e:
    #     return f"An unexpected error occurred: {e}"
    
# def load_scaler(path):
#     return joblib.load(path)

# def load_model_predict(model_choice,model_path,X):
#     if model_choice == 'xgb':
#         model = joblib.load(model_path)
#         return model.predict(X)
#     elif model_choice == 'pr':
#         # Load and predict with Polynomial Regression model
#         model = mlflow.sklearn.load_model(model_path)
#         return model.pred(X,True)
#     else:
#         return "Please select a valid model."

# def get_scaler_path():
#     return os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# def get_model_path():
#     model_path = os.path.join(BASE_DIR, '..', 'models', 'model.pkl')
#     # poly_model_path = os.path.join(BASE_DIR, '..', 'models', 'model_polynomial.pkl')
#     model_uri = "../mlruns/0/models/m-8f5af5f6d6404e678c6762691bd8faee/artifacts"

#     return model_path,model_uri