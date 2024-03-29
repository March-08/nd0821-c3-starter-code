# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used in this project is a RandomForest.
The best hyperparameters found are the following:
{'max_depth': 12, 'min_samples_split': 45, 'n_estimators': 20}

## Intended Use

This model should be used to predict if a person earns more or less than 50k.

## Training Data

The model was trained on the census.csv dataset. This dataset was made available to students from the Udacity professors. For more information about the dataset contact me.

## Evaluation Data

The dataset was randomly split leaving 20% of it for evaluatin pourposes.

## Metrics

The metrics on which I have focused are: precision, recall and fbeta.
Results:
{'precision': 0.7972972972972973, 'recall': 0.5633354551241248, 'fbeta': 0.6602014173815741}

## Ethical Considerations

The model has been tested on several slices of the dataset to mitigate as much as possible any kind of biases.

## Caveats and Recommendations

Please consider that this repository could cointains bugs. It still needs some work.
