from flask import Flask, request
from flask_restful import Resource, Api
from joblib import load
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
MODEL = ROOT / 'model'

app = Flask(__name__)
api = Api(app)

class Classifier(Resource):
    
    def __init__(self):
        self.clf = load(MODEL/'clf.joblib')

    def get(self):
        sepal_length = request.args.get('sepal_length', None)
        sepal_width = request.args.get('sepal_width', None)
        petal_length = request.args.get('petal_length', None)
        petal_width = request.args.get('petal_width', None)

        x = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]]
            , dtype=np.float64)

        y = self.clf.predict(x)
        y_prob = self.clf.predict_proba(x)

        return {'class': y.item(), 'confidance': y_prob.max()}
    

api.add_resource(Classifier, '/predict')

if __name__ == '__main__':
    app.run(debug=True)