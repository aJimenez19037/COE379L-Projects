from flask import Flask, request, jsonify
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
import requests
import json



with open('models/acc_xgb','rb') as f:
    acc_xgb = pickle.load(f)

with open('models/recall_xgb','rb') as f1:
    recall_xgb = pickle.load(f1)

with open('models/acc_svm', 'rb') as f2:
    acc_svm = pickle.load(f2)

with open('models/recall_svm', 'rb') as f3:
    recall_svm = pickle.load(f3)


app = Flask(__name__)

@app.route('/models', methods = ["POST"])
def get_data():
    global gdata
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    data = request.files['file']
    if data.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    data = pd.read_csv(data)
    gdata = data
    return ('Data loaded\n')



@app.route('/models/acc_xgb', methods = ['GET'])
def model_info_acc_xgb():
    return{
        "name": "acc_xgb",
        "type": "xgboost",
        "optimized for": "accuracy",
        "accuracy score on test": "0.70",
        "true recall score on test": "0.22"
    }
@app.route('/models/acc_xgb', methods=['POST'])
def classify_acc_xgb():
    d = acc_xgb.predict(gdata).tolist()
    bool_list = list(map(bool,d))
    ##return jsonify(acc_xgb.predict(gdata).tolist())
    return jsonify(bool_list)

@app.route('/models/recall_xgb', methods = ['GET'])
def model_info_recall_xgb():
    return{
        "name": "recall_xgb",
        "type": "xgboost",
        "optimized for": "recall",
        "accuracy score on test": "0.72",
        "true recall score on test": "0.26"
                                            }
@app.route('/models/recall_xgb', methods=['POST'])
def classify_recall_xgb():
    d = recall_xgb.predict(gdata).tolist()
    bool_list = list(map(bool,d))
    return jsonify(bool_list)


@app.route('/models/acc_svm', methods = ['GET'])
def model_info_acc_svm():
    return{
        "name": "acc_svm",
        "type": "svm",
        "optimized for": "accuracy",
        "accuracy score on test": "0.67",
        "true recall score on test": "0.09"
                                            }
@app.route('/models/acc_svm', methods=['POST'])
def classify_acc_svm():
    return jsonify(acc_svm.predict(gdata).tolist())

@app.route('/models/recall_svm', methods = ['GET'])
def model_info_recall_svm():
    return{
        "name": "recall_svm",
        "type": "svm",
        "optimized for": "recall",
        "accuracy score on test": "0.63",
        "true recall score on test": "0.26"
        }
@app.route('/models/recall_svm', methods=['POST'])
def classify_recall_svm():
    return jsonify(recall_svm.predict(gdata).tolist())
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

