import requests
import pytest

def test_predict():
    params = {"sepal_length": 5.1,"sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = requests.get('http://127.0.0.1:5000/predict', params=params)
    data = response.json()

    assert data['class'] == "setosa" and \
           data['confidance'] == pytest.approx(0.97, 0.1)
