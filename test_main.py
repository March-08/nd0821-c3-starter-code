import json
from fastapi.testclient import TestClient
from main import app, Data

client = TestClient(app)

example_1 = {
    "age": 62,
    "workclass": "Private",
    "fnlgt": 57346,
    "education": "Doctorate",
    "education_num": 82,
    "marital_status": "Never-married",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 140,
    "capital_loss": 0,
    "hours_per_week": 67,
    "native_country": "United-States",
}

example_2 = {
    "age": 24,
    "workclass": "State-gov",
    "fnlgt": 584421,
    "education": "Bachelors",
    "education_num": 14,
    "marital_status": "Separated",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 3,
    "capital_loss": 2,
    "hours_per_week": 24,
    "native_country": "United-States",
}


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.text == "Welcome to my Udacity solution!"


def test_post_1():
    url = "/inference"  # Use the relative URL for the endpoint

    data = {
        "age": 62,
        "workclass": "Private",
        "fnlgt": 57346,
        "education": "Doctorate",
        "education_num": 82,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 140,
        "capital_loss": 0,
        "hours_per_week": 67,
        "native_country": "United-States",
    }
    headers = {"Content-Type": "application/json"}
    response = client.post(url, json=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == ">50K"


def test_post_2():
    data = json.dumps(example_2)
    headers = {"Content-Type": "application/json"}
    response = client.post("/inference", data=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == "<=50K"
