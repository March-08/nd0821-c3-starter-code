"""
Unit test of model.py module with pytest
author: Marcello Piliti
"""

import pytest, os, logging, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import os.path
import numpy as np
from ml.model import (
    inference,
    compute_model_metrics,
)
from ml.data import process_data
import joblib


"""
Fixture 
"""


@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    datapath = "./data/census.csv"
    return pd.read_csv("./data/census.csv", sep=", ", engine="python")


@pytest.fixture(scope="module")
def predictions():
    # code to load in the data.
    predictions_path = "./model/predictions.npy"
    loaded_predictions = np.load(predictions_path)

    return loaded_predictions


@pytest.fixture(scope="module")
def processed_data(data):
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, *_ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def path():
    return "./data/census.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features


"""
Test methods
"""


def test_train_model(path):
    try:
        assert os.path.isfile(path)
    except FileNotFoundError as e:
        logging.error(str(e))


def test_compute_model_metrics(processed_data, predictions):
    _, _, _, y_test = processed_data
    assert len(y_test) == len(predictions)


def test_inference(processed_data):
    # loaded_model = pickle.load(open("model/model.pkl", "rb"))
    loaded_model, encoder, lb = joblib.load("./model/model.pkl")
    _, _, X_test, y_test = processed_data
    preds = inference(loaded_model, X_test)
    assert isinstance(preds, np.ndarray)  # Check if output is numpy array
    assert preds.shape[0] == y_test.shape[0]
