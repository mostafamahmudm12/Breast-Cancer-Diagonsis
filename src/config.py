import os
from  dotenv import load_dotenv
import joblib

# Get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Parent folder path
SRC_FOLDER_PATH=os.path.dirname(os.path.abspath(__file__))

# Load Model
MODEL_KNN=joblib.load(os.path.join(SRC_FOLDER_PATH,'assets','KNN_model.pkl'))
MODEL_LOGISTIC=joblib.load(os.path.join(SRC_FOLDER_PATH,'assets','logistic_regression_model.pkl'))
MODEL_Random=joblib.load(os.path.join(SRC_FOLDER_PATH,'assets','Random_model.pkl'))
MODEL_SVM=joblib.load(os.path.join(SRC_FOLDER_PATH,'assets','SVM_model.pkl'))
SCALER=joblib.load(os.path.join(SRC_FOLDER_PATH,'assets','scaler.pkl'))