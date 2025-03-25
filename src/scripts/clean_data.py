import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("./athlete_events.csv")

    df["Medal_won"] = df["Medal"].apply(lambda x: 0 if pd.isna(x) or x == "NA" else 1)

    for col in ["Age", "Height", "Weight"]:
        df[col].fillna(df[col].median(), inplace=True)

    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

    df_clean = df[["Sex", "Age", "Height", "Weight", "Sport", "Medal_won"]].copy()

    df_clean = pd.get_dummies(df_clean, columns=["Sport"], drop_first=True)

    df_clean.to_csv("athlete_events_cleaned.csv", index=False)
    
if __name__ == "__main__":
    main()