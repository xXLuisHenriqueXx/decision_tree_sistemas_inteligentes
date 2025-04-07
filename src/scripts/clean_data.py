import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("./athlete_events.csv")

    # Seleciona as colunas de medalha e remove as linhas com medalhas NA, inserindo 0 para medalhas não ganhas e 1 para medalhas ganhas
    df["Medal_won"] = df["Medal"].apply(lambda x: 0 if pd.isna(x) or x == "NA" else 1)

    # Preenche os valores ausentes de "Age", "Height" e "Weight" com a mediana de cada coluna
    for col in ["Age", "Height", "Weight"]:
        df[col].fillna(df[col].median(), inplace=True)

    # Converte os valores da coluna para binário
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

    # Seleciona apenas as linhas onde o esporte é "Swimming"
    df = df[df["Sport"] == "Swimming"]

    # Copia as colunas relevantes para um novo DataFrame e remove as colunas desnecessárias
    df_clean = df[["Sex", "Age", "Height", "Weight", "Team", "Medal_won"]].copy()

    # Converte a coluna "Team" em variáveis dummy (0 e 1), aplicando o one-hot encoding
    df_clean = pd.get_dummies(df_clean, columns=["Team"], drop_first=True)

    # Cria o novo arquivo CSV com os dados limpos
    df_clean.to_csv("athlete_events_cleaned.csv", index=False)
    
if __name__ == "__main__":
    main()