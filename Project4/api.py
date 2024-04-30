from flask import Flask, request
import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

with open('acc_xgb','rb') as f:
    acc_xgb = pickle.load(f)

with open('recall_xgb','rb') as f1:
    recall_xgb = pickle.load(f1)

app = Flask(__name__)

@app.route('/models/acc_xgb', methods = ['GET'])
def model_info_acc_xgb():
    return{
        "name": "acc_xgb",
        "type": "xgboost",
        "optimized for": "accuracy"
    }
@app.route('/models/acc_xgb', methods=['POST'])
def classify_acc_xgb():
    entry = request.json('entry')
    if not entry:
        return {"error": "The 'entry' field is required"}, 404
    return {"result":acc_xgb.predict(data)}
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
