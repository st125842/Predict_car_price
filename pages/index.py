from dash import Dash, html, Output, Input, dcc
import dash
import dash_bootstrap_components as dbc
import os
# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, use_pages=True,pages_folder=os.path.dirname(os.path.abspath(__file__)), external_stylesheets=[dbc.themes.BOOTSTRAP],)
# from main_page import *

# Navigation Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("XGBoost", href="/model1")),
    ],
    brand="Well Come To My page ST125842",
    brand_href="/",
    color="primary",
    dark=True,
)


app.layout = html.Div([
    navbar,
    dash.page_container
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)