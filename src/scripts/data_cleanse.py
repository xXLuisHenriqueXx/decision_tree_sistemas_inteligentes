import pandas as pd
import numpy as np

def main():
    # 1. Leitura do dataset original.
    # O arquivo athlete_events.csv está localizado na pasta ../data (relativo a esse script).
    df = pd.read_csv("../data/athlete_events.csv")

    # 2. Criação da coluna 'Medal_won'.
    # Se o atleta ganhou alguma medalha (Gold, Silver ou Bronze), atribui 1; caso contrário, atribui 0.
    df["Medal_won"] = df["Medal"].apply(lambda x: 0 if pd.isna(x) or x == "NA" else 1)

    # 3. Preenchimento de valores ausentes.
    # Para garantir um tratamento consistente, as colunas numéricas "Age", "Height" e "Weight"
    # têm seus valores ausentes substituídos pela mediana de cada coluna.
    for col in ["Age", "Height", "Weight"]:
        df[col].fillna(df[col].median(), inplace=True)

    # 4. Conversão da coluna 'Sex' para numérico.
    # Mapeia "M" para 1 e "F" para 0 para facilitar o processamento por algoritmos de machine learning.
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

    # 5. Seleção das colunas relevantes.
    # São escolhidas colunas que irão servir como features para a árvore de decisão, além da variável alvo.
    df_clean = df[["Sex", "Age", "Height", "Weight", "Sport", "Medal_won"]].copy()

    # 6. Transformação da variável categórica 'Sport'.
    # Aplica o one-hot encoding para transformar a coluna 'Sport' em múltiplas colunas dummy (=0 ou 1),
    # descartando a primeira categoria com drop_first para evitar correlação entre as variáveis.
    df_clean = pd.get_dummies(df_clean, columns=["Sport"], drop_first=True)

    # 7. Exportação do DataFrame limpo para um novo arquivo CSV.
    df_clean.to_csv("athlete_events_cleaned.csv", index=False)
    
if __name__ == "__main__":
    main()