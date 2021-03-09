from flask import Flask, request
import pandas as pd
import json
import pickle
import APIHelper




def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/predict', methods=['GET'])
    def predict():
        data = APIHelper.load_request(request)
        model = APIHelper.load_model()
        result = APIHelper.make_prediction(model, data)
        return result

    return app
