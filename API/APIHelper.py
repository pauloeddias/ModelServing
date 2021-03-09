import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import json


def load_request(request):
    data = request.get_json()
    data = pd.DataFrame(data)
    return data


def load_model() -> DecisionTreeClassifier:
    f = open('Training/model.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    return model


def make_prediction(model: DecisionTreeClassifier, data: pd.DataFrame):
    pred = model.predict(data).tolist()
    ret = json.dumps(pred)
    return ret
