import joblib
import os 
import mlflow
# from models.model import Polynomial,LinearRegression

def load_scaler(path):
    return joblib.load(path)

# def load_model_predict(model_choice,model_path,X):
#     if model_choice == 'xgb':
#         model = joblib.load(model_path)
#         return model.predict(X)
#     elif model_choice == 'pr':
#         # Load and predict with Polynomial Regression model
#         model = joblib.load(model_path)
#         # model = mlflow.sklearn.load_model(model_path)
#         return model.pred(X,True)
#     else:
#         return "Please select a valid model."

def get_scaler_path():
    return 'models\scaler.pkl'

def get_model_path():
    model_path = 'models\model.pkl'
    # poly_model_path = os.path.join(BASE_DIR, '..', 'models', 'model_polynomial.pkl')
    # model_uri = "../mlruns/0/models/m-8f5af5f6d6404e678c6762691bd8faee/artifacts"
    model_uri = 'models\model_polynomial.pkl'

    return model_path,model_uri