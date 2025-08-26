import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import os
from xgboost import XGBRegressor

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash.register_page(__name__, path='/model1')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

text = html.Div([
    html.H1("Prediction Page"),
])
# Creating FORM
year = html.Div(
    [
        dbc.Label("year", html_for="example-email"),
        dbc.Input(id="year", type="number", placeholder=""),
        dbc.FormText(
            "This is the value for year feature",
            color="secondary",
        ),
    ],
    className="mb-3",
)

engine = html.Div(
    [
        dbc.Label("engine", html_for="example-email"),
        dbc.Input(id="engine", type="number", placeholder=""),
        dbc.FormText(
            "This is the value for engine feature",
            color="secondary",
        ),
    ],
    className="mb-3",
)

max_power = html.Div(
    [
        dbc.Label("max_power", html_for="example-email"),
        dbc.Input(id="max_power", type="number", placeholder=""),
        dbc.FormText(
            "This is the value for max_power feature",
            color="secondary",
        ),
    ],
    className="mb-3",
)


seats = html.Div(
    [
        dbc.Label("seats", html_for="example-email"),
        dbc.Input(id="seats", type="number", placeholder=""),
        dbc.FormText(
            "This is the value for seats feature",
            color="secondary",
        ),
    ],
    className="mb-3",
)

fuel = html.Div(
    [
        dbc.Label("fuel", html_for="example-email"),
        dbc.Input(id="fuel", type="number", placeholder=""),
        dbc.FormText(
            "This is the value for fuel feature (0 for Diesel and 1 for Petrol)",
            color="secondary",
        ),
    ],
    className="mb-3",
)

tranmission = html.Div(
    [
        dbc.Label("tranmission", html_for="example-email"),
        dbc.Input(id="tranmission", type="number", placeholder=""),
        dbc.FormText(
            "This is the value for tranmission feature (0 for Auto and 1 for Manual)",
            color="secondary",
        ),
    ],
    className="mb-3",
)


submit_hardcode = html.Div([
            dbc.Button(id="submit_hardcode", children="Submit", color="primary", className="me-1"),
            dbc.Label("price is: "),
            html.Output(id="y_hardcode", children="")
], style={'marginTop':'10px'})


form =  dbc.Form([
            year,
            engine,
            max_power,
            seats,
            fuel,
            tranmission,
            submit_hardcode,
        ],
        className="mb-3")


layout =  dbc.Container([
        text,
        form,
    ], fluid=True)

@callback(
    Output(component_id="y_hardcode", component_property="children"),
    State(component_id="year", component_property="value"),
    State(component_id="engine", component_property="value"),
    State(component_id="max_power", component_property="value"),
    State(component_id="seats", component_property="value"),
    State(component_id="fuel", component_property="value"),
    State(component_id="tranmission", component_property="value"),
    Input(component_id="submit_hardcode", component_property='n_clicks'),
    prevent_initial_call=True
)
def result(year,engine,max_power,seats,fuel,tranmission,submit_hardcode):
    print(year,engine,max_power,seats,fuel,tranmission,submit_hardcode)
    if year == None:
        year = 2020
    if max_power == None:
        max_power = 92.15565094167124
    if engine == None:
        engine = 1248.0
    if seats == None:
        seats = 5.0
    if fuel == None:
        fuel = 0
    if tranmission == None:
        tranmission = 1 
    print(os.getcwd())
    # print(year + engine)
    # model = XGBRegressor()
    # model.load_model("..\xgb_model.json")
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    X =[[year,engine,max_power,seats,fuel,tranmission]]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    pred = np.exp(pred)
    print(pred)
    return str(pred[0])

