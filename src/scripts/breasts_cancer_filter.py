import pandas as pd
import requests
from io import StringIO
import os

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

columns = [
    "ID", "Diagnosis",
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "SE Radius", "SE Texture", "SE Perimeter", "SE Area", "SE Smoothness",
    "SE Compactness", "SE Concavity", "SE Concave Points", "SE Symmetry", "SE Fractal Dimension",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

response = requests.get(data_url)
data = response.text

data_frame = pd.read_csv(StringIO(data), header=None, names=columns)

output_file = "./src/data/breast_cancer_wisconsin_diagnostic.csv"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

data_frame.to_csv(output_file, index=False)