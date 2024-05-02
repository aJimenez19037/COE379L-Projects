import requests


rsp = requests.post("http://172.17.0.1:5000/models/acc_xgb", json={"entry":1})

print(rsp.json())
