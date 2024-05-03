# Purpose
Store and submit all 4 projects for COE379 - Software Design for Responsible Intelligent Systems

## Project 1 - Fuel Efficiency Linear Regression Model 
The purpose of this project was to create a linear regression model based on least squared method. Throughout this project I prepared and analyzed the data, split the data into a training and test set, fit the data to the linear regression model, and evaluated its accuracy. In the end we ended up with an R2 score of 0.834. More detail can be foud the in report (.pdf).

## Project 2 - Breast Cancer Recurrence Classification Models
The purpose of this project was to determine the best model for predicting if a patient was likely to have a recurrence event for breast cancer. In this project 3 models were built: Random Forest, K-Neares Neighbor, and Multinomial Naive Bayes. Throughout this project I prepared and analyzed the data, split the data into training and test set, fit the data to the different models, and evaluated their performance using different metrics. The models in this project were optimized for maximizing recall using sklearn grid search function. More detail can be found in the report (.pdf).

## Project 3 - Hurricane Harvey Classification Inference Server
This purpose of this project was to create an inference server that wraps a model designed to classify satellite images with damage or no damage taken from a satellite from Hurricane Harvey. Through this project several dense/ANN models were built a long with a lenet5 based model and an alternate lenet5 model based on the paper: ["Building Damage Annotation on Post-Hurricane.
Satellite Imagery Based on Convolutional Neural
Networks"](https://arxiv.org/pdf/1807.01688.pdf). This project involved Data preprocessing and visualization, model design, training, and evaluation, and model deployment. 

### Files
images-cnn-split - contains images split into test, train and further split into damage and no damage which is used when training the model.

images - contains all of the images provided

keras models - contains saved CNN and ANN models models.

Dockerfile - used to build docker container to run flask app

HurricaneDamageANN.ipynb - jupyter notebook used to train ANN models

HurricaneDamageCNN.ipynb - jupyter notebook used to train CNN models

api.py - flask application

test_api.py - python script used to make a request to flask application

### Using the Inference Server
This is written under the assumption you are using an ssh tunnel developed in class. To close the server simply press ctrl-c in the terminal. The following commands will launch the flask app:
```
docker pull antjim19037/hurricane_api 
docker run -it --rm -p 5000:5000 antjim19037/hurricane_api
```
In another terminal clone this repository into your vm 
```
git clone git@github.com:aJimenez19037/COE379L-Projects.git
mkdir nb-data
cp COE379L-Projects/Project3/ ~/nb-data
```
Go into class container
```
docker pull jstubbs/coe379l
docker run --rm --name nb -p 8888:8888 -v $(pwd)/nb-data:/code -d --entrypoint=sleep jstubbs/coe379l infinity
docker exec -it nb bash
```
Running test_api.py - First step is to install tensorflow_datasets as the class container does not install install. You can then run the python script by passing in an image. In the commands below we pass a test image that is labeled as damage. We see the model output agrees as the probability that it is damage is 0.9999943971633911. Additionally this script is written to use the alt_lenet5 route as it performed the best out of all of the models built for this project. 
```
pip install tensorflow_datasets
cd Project3/
python3 test_api.py images-cnn-split/test/damage/-93.53950999999999_30.982944.jpeg

{'result': [[0.9999943971633911, 5.603660611086525e-06]]}
```
### Flask application usage
Below are the routes in the flask application. The ANN models were not included as they performed poorly compared to the CNN models. Additionally, lenet5 model has an accuracy around 0.96 and the alternate lenet5 model has an accuracy around 0.97. Additional notes about the inference server is that it only works when passing in one image at a time. The image must be 128x128x3 and does not need to be preprocessed. The Flask server performs preprocessing on the inputted image. 
Here is a table of the routes:

| Route | Method | Description |
| --- | --- | --- |
| `/models/alt_lenet5` | POST | Returns classification of the image using alt_lenet5 model|
| `/models/alt_lenet5` | GET | Return information of the alt_lenet5 model |
| `/models/lenet5` | POST | Returns classification of the image using lenet5 model|
| `/models/lenet5` | GET | Return information of the lenet5 model |

## Project 4 - SVM and XGBoost Model Exploration on Breast Cancer Recurrence Data
The purpose of this project was to examine how models using SVM and XGBoost might improve or change our results from the breast cancer recurrence dataset that we used in Project 2. We created two of each type of model, one that optimized recall and one that optimized accuracy. We ultimately had four models that we included in our inference server.
### Files
Dockerfile - used to build docker container to run flask app

SVM.ipynb - jupyter notebook used to train SVM models

XGBoost.ipynb - jupyter notebook used to train XGBoost models

api.py - flask application

models - contains saved SVM and XGBoost models

project4.csv - our original, unprocessed data

testdata.csv - our processed, test data

### Using the Inference Server
This is written under the assumption you are using an ssh tunnel developed in class. To close the server simply press ctrl-c in the terminal. The following commands will launch the flask app:
Open a terminal and use the ssh tunnel. Then, execute the following commands
```
docker pull avlavelle/starfish 
docker run -it --rm -p 5000:5000 avlavelle/starfish
```
In another terminal, clone this repository onto your vm and move into the Project4 folder.
```
git clone git@github.com:aJimenez19037/COE379L-Projects.git
cd Project4
```
Now you are ready to use the ```POST``` route of the flask application to load the data. In your commandline, execute
```
curl -X POST -F "file=@testdata.csv" 127.0.0.1:5000/models
```
If successful, ```Data loaded``` will be printed. This route can be used to load in any version of the data that is processed according to what is found in our ```.ipynb``` files. If your file uses a different name than ```testdata.csv```, be sure to replace the file name in the curl command and be sure that it is located in the Project 4 directory. This ```POST``` ```/models``` route must be curled before any of the other routes are called in order to load in the data.

### Flask application usage
Below are the routes in the flask application. Four models were included in the application. We included two of each model type, one optimized for recall and one optimized for accuracy, for both XGBoost and SVM. The Flask server requires the user to first ```POST``` a preprocessed ```.csv``` file to the ```/models``` route to load in the data. The model specific ```GET``` routes provide information on the models, such as their accuracy and recall scores. The model specific ```POST``` routes provide the predictions of breast cancer recurrence for the loaded in data.
Here is a table of the routes:

| Route | Method | Description |
| --- | --- | --- |
| `/models` | POST | Loads in the data from the provided ```.csv``` file|
| `/models/acc_xgb` | GET | Returns information of the acc_xgb model |
| `/models/acc_xgb` | POST | Returns classifications of the data using acc_xgb model|
| `/models/recall_xgb` | GET | Returns information of the recall_xgb model |
| `/models/recall_xgb` | POST | Returns classifications of the data using recall_xgb model|
| `/models/acc_svm` | GET | Returns information of the acc_svm model |
| `/models/acc_svm` | POST | Returns classifications of the data using acc_svm model|
| `/models/recall_svm` | GET | Returns information of the recall_svm model |
| `/models/recall_svm` | POST | Returns classifications of the data using recall_svm model|

### Sample Routes and Outputs
Demonstrated below are exactly how to call the routes and their sample outputs. Use the above table to curl different models.
Route:
```
curl -X POST -F "file=@testdata.csv" 127.0.0.1:5000/models
```
Output:
```
Data loaded
```
Route:
```
curl -X GET 127.0.0.1:5000/models/recall_svm
```
Output:
```
{
  "accuracy score on test": "0.63",
  "name": "recall_svm",
  "optimized for": "recall",
  "true recall score on test": "0.26",
  "type": "svm"
}
```
Route:
```
curl -X POST 127.0.0.1:5000/models/recall_svm
```
Output:
```
[
  false,
  true,
  ...,
  ...,
  false
]
```













