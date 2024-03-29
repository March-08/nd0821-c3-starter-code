import json
from fastapi.testclient import TestClient
from main import app, Data

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.text == "Welcome to my Udacity solution!"


def test_post_1():
    data = json.dumps(Data.model_config["json_schema_extra"]["examples"][0])
    r = client.post("/inference", data=data)
    assert r.status_code == 200
    assert r.text == ">50K"


def test_post_2():
    data = json.dumps(Data.model_config["json_schema_extra"]["examples"][1])
    r = client.post("/inference", data=data)
    assert r.status_code == 200
    assert r.text == "<=50K"
