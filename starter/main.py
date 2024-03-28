# Put the code for your API here.
from ml.model import train_model
import pandas as pd

df = pd.read_csv("data/census.csv")
print(df)
