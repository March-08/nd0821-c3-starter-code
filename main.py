from fastapi import FastAPI, Body, Response, status
from pydantic import BaseModel
from ml.model import inference
from ml.data import process_data
import logging
from pydantic import BaseModel
import os
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO)

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                },
                {
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
                },
            ]
        }
    }


@app.get("/")
def read_root():
    response = Response(
        status_code=status.HTTP_200_OK,
        content="Welcome to my Udacity solution!",
    )
    return response


model, encoder, lb = joblib.load("./model/model.pkl")
cat_features = [f for (f, t) in Data.__annotations__.items() if t == str]


# model inference:
@app.post("/inference")
def predict(data: Data):
    try:
        data = pd.DataFrame(data.__dict__, [0])

        data, *_ = process_data(
            data, categorical_features=cat_features, training=False, encoder=encoder
        )

        pred = inference(model, data)
        to_ret = lb.inverse_transform(pred)[0]

        return Response(
            status_code=status.HTTP_200_OK,
            content=to_ret,
        )
    except Exception as e:
        return Response(
            status_code=status.HTTP_200_OK,
            content=f"something didn't work: {str(e)} ",
        )
