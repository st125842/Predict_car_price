import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = dbc.Container([
    html.H1("Welcome to the Car Price Predictor", className="text-center my-4 text-primary"),
    html.Hr(className="my-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Model Usage Instructions", className="text-info mb-3"),
                    html.P("Welcome! Use this page to understand how to use the prediction models available in this app."),
                    html.P("To get started, navigate to the 'Predict' page to input your car's specifications and see a price prediction."),
                ])
            ], className="mb-4 shadow-sm")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📌 Available Models", className="text-info mt-1 mb-3"),
                    dbc.Accordion([
                        dbc.AccordionItem([
                            html.P("XGBoost is a powerful ensemble method based on decision trees. It builds models sequentially where each new tree corrects the previous one's errors."),
                            html.Ul([
                                html.Li("🔍 Type: Supervised Learning"),
                                html.Li("💡 Uses: Handling structured/tabular data efficiently"),
                                html.Li([
                                    "🛠 Steps:",
                                    html.Ol([
                                        html.Li("Initialize a base model"),
                                        html.Li("Compute residual errors"),
                                        html.Li("Fit a decision tree to residuals"),
                                        html.Li("Update the model (weighted by learning rate)"),
                                        html.Li("Repeat until stopping criteria")
                                    ])
                                ])
                            ])
                        ], title="📈 XGBoost (Extreme Gradient Boosting)", className="mb-2"),

                        dbc.AccordionItem([
                            html.P("Polynomial Regression fits a curved line (polynomial function) to the data. It's useful when relationships between variables are non-linear."),
                            html.Ul([
                                html.Li("🔍 Type: Supervised Learning"),
                                html.Li("📐 Fits polynomial functions (e.g., quadratic, cubic)"),
                                html.Li("💡 Use for modeling curves in data"),
                                html.Li([
                                    "🛠 Steps:",
                                    html.Ol([
                                        html.Li("Choose the polynomial degree (e.g., 2 for quadratic)"),
                                        html.Li("Transform input features to polynomial terms"),
                                        html.Li("Fit a Linear Regression on transformed features"),
                                        html.Li("Make predictions")
                                    ])
                                ])
                            ])
                        ], title="📉 Polynomial Regression", className="mb-2"),
                    ])
                ])
            ], className="mb-4 shadow-sm")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔮 Prediction Input Requirements", className="bg-info text-white"),
                dbc.CardBody([
                    html.P("To make a prediction, please provide the following 6 features:", className="lead"),
                    html.Ul([
                        html.Li("📅 Year: Numeric input (e.g., 2020)"),
                        html.Li("⚙️ Engine: Numeric input (e.g., 1245 for 1245cc)"),
                        html.Li("⚡ Max Power: Numeric input (e.g., 100 for 100 bhp)"),
                        html.Li("🪑 Seats: Numeric input (e.g., 5)"),
                        html.Li("⛽ Fuel: Use 0 for Diesel, 1 for Petrol"),
                        html.Li("🔁 Transmission: Use 0 for Auto, 1 for Manual")
                    ], className="list-unstyled"),
                    html.Hr(),
                    html.P("If a field is left empty, the system will auto-fill with default values:", className="fw-bold text-muted"),
                    html.Ul([
                        html.Li("📅 Year: 2020"),
                        html.Li("⚙️ Engine: Median from training data"),
                        html.Li("⚡ Max Power: Mean from training data"),
                        html.Li("🪑 Seats: Most common value"),
                        html.Li("⛽ Fuel: Most common fuel type"),
                        html.Li("🔁 Transmission: Most common transmission type")
                    ], className="text-muted small")
                ])
            ], className="mb-4 shadow-sm")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Alert("✅ Ready to try it out? Navigate to the Predict page!", color="success", className="text-center my-4")
        ])
    ])
], fluid=False, className="my-5")
