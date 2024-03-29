# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    compute_slice_performances,
)
import pandas as pd
import joblib
import logging
import numpy as np

logging.basicConfig(
    filename="results.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

data = pd.read_csv("./data/census.csv", sep=", ", engine="python")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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


# Train and save a model.

model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
np.save("model/predictions.npy", preds)
results = {"precision": precision, "recall": recall, "fbeta": fbeta}

logging.info(f"train results: {results}")
joblib.dump((model, encoder, lb), "./model/model.pkl")

# save results
with open("./results.txt", "w+") as f:
    txt = str(model.best_params_) + "\n" + str(results)
    f.write(txt)

# compute slise metrics
slice_metrics = compute_slice_performances(
    test, cat_features, "marital-status", model, encoder, lb
)
