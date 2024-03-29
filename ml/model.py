from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging
from ml.data import process_data
import pandas as pd

logging.basicConfig(
    filename="results.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    parameters = {
        "n_estimators": [20, 30, 40],
        "max_depth": [6, 12],
        "min_samples_split": [15, 45, 95],
    }

    clf = GridSearchCV(
        RandomForestClassifier(
            random_state=0,
        ),
        param_grid=parameters,
        cv=3,
        verbose=2,
    )

    clf.fit(X_train, y_train)
    logging.info("********* Best parameters found ***********")
    logging.info("BEST PARAMS: {clf.best_params_}")

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_slice_performances(df, cat_features, slice_feature, model, encoder, lb):
    """
    Computes metrics for each slice of data that has a particular value for a given feature.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe on which to work.
    cat_features : list[str]
        List wth names of the categorical features on the dataframe.
    slice_feature : str
        Feature name to compute the slice permormance metrics  for.
    model : ???
        Trained ml model to use.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    slice_metrics : dict
        Dictionary containing the sliced performances
    """
    slice_metrics = {}
    for value in df[slice_feature].unique():
        X_slice = df[df[slice_feature] == value]
        X_slice, y_slice, _, _ = process_data(
            X_slice,
            cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slice_metrics[value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }
        logging.info(
            f"sliced metrics of {slice_feature} = {value}: {slice_metrics[value]}"
        )

    # write to slice_output.txt
    with open("slice_output.txt", "w") as f:
        for key, value in slice_metrics.items():
            f.write(f"{key}: {value}")
            f.write("\n")
    return slice_metrics


# def make_single_prediction(input_json, model_dir):
#     """Make a prediction using a trained model.

#     Inputs
#     ------
#     input_json : dict
#         Input data in json format.
#     model_dir : str
#         Path to the directory containing the trained model, encoder & lb.
#     Returns
#     -------
#     preds : str
#         Outpurs the model predictions.
#     """
#     # Convert the input to a dataframe with same data types as training data.
#     input_df = pd.DataFrame(dict(input_json), index=[0])
#     logging.info(f"input_df: {input_df}")

#     # clean data
#     cleaned_df, cat_cols, num_cols = basic_cleaning(
#         input_df, "data/census_cleaned.csv", "salary", test=True
#     )

#     # load model, encoder, and lb and predict on single json instance
#     model, encoder, lb = load_model(model_dir)

#     # process data
#     X, _, _, _ = process_data(
#         cleaned_df, cat_cols, training=False, encoder=encoder, lb=lb
#     )

#     # predict
#     preds = inference(model, X)
#     preds_class = lb.inverse_transform(preds)

#     return {preds_class[0]}
