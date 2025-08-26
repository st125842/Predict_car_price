import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/')

layout = html.Div([
    html.H2("Instruction How To Should My Model"),
    html.P("You can click XGBoost predict on navigation bar to predict"),
    html.Div([
        html.P("I use XGBoosting Model for this task.I will briefly what is XGBoosting"),
        html.H4("XGBoost (Extreme Gradient Boosting)"),
        html.Ul([
            html.Li("Type: Supervised learning algorithm"),
            html.Li("Core Idea: Gradient boosting — builds an ensemble of decision trees sequentially, where each new tree tries to correct errors of previous trees."),
            html.Li("Steps"),
            html.Ol([html.Li("Initialize a model"),
                     html.Li("Compute residuals"),
                     html.Li("Fit a new tree to predict"),
                     html.Li("Update the model (weighted by learning rate)"),
                     html.Li('Repeat until max trees or early stopping.')
                     ]),
        ])    
    ]),

    html.Div([
        html.H4('Prediction'),
        html.P('For prection page required 6 freature'),
        html.Ul([
            html.Li("Years: input only numeric, example 2020"),
            html.Li("engine: input only numeric, example  1245 cc then type 1245"),
            html.Li("max power: input only numeric, example 100 bhp then type 100"),
            html.Li("seats: input only numeric, example 5"),
            html.Li("fuel:  input only numeric 0 for Diesel and 1 for Petrol "),
            html.Li("transmission: input only numeric 0 for Auto and 1 for Manual"),
        ]),
        html.P('For Unknow value you can skip that field but I will fill that value by'),
        html.Ul([
            html.Li("Years: I will fill 2020 that is latest year."),
            html.Li("engine: I will fill median base on my training set"),
            html.Li("max power: I will fill mean base on my training set"),
            html.Li("seats: I will fill mode base on my training set"),
            html.Li("fuel: I will fill mode base on my training set"),
            html.Li("transmission: I will fill mode base on my training set"),
        ]),

    ])
])
