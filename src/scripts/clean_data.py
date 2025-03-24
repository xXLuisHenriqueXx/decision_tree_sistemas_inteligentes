import pandas as pd
import requests
from io import StringIO

def load_data(url):
    response = requests.get(url)
    data = response.text
    return pd.read_csv(StringIO(data), header=None)

def preprocess_data(df):
    # Assign proper column names
    columns = [
        "ID", "Diagnosis",
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
        "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
        "SE Radius", "SE Texture", "SE Perimeter", "SE Area", "SE Smoothness",
        "SE Compactness", "SE Concavity", "SE Concave Points", "SE Symmetry", "SE Fractal Dimension",
        "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
        "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
    ]
    df.columns = columns

    # Drop ID column (not useful for prediction)
    df.drop(columns=["ID"], inplace=True)

    # Convert diagnosis to binary (M=1 for malignant, B=0 for benign)
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

    # Drop rows with missing values (if any)
    df_cleaned = df.dropna()

    return df_cleaned

def main():
    # URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

    # Load and clean data
    df = load_data(url)
    df_cleaned = preprocess_data(df)

    # Save the cleaned dataset
    output_file = "breast_cancer_cleaned.csv"
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved as {output_file}")

if __name__ == "__main__":
    main()
