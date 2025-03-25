import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def load_data(filepath="athlete_events_cleaned.csv"):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop("Medal_won", axis=1)
    y = df["Medal_won"]
    return X, y

def create_subset_by_ratio(df, non_medal_ratio):
    df_medal = df[df["Medal_won"] == 1]
    df_non_medal = df[df["Medal_won"] == 0]
    n_medal = len(df_medal)
    target_non_medal = int((non_medal_ratio / (1 - non_medal_ratio)) * n_medal)
    if target_non_medal > len(df_non_medal):
        target_non_medal = len(df_non_medal)
    df_non_sampled = df_non_medal.sample(n=target_non_medal, random_state=0)
    df_subset = pd.concat([df_medal, df_non_sampled]).sample(frac=1, random_state=0).reset_index(drop=True)
    return df_subset

def tune_hyperparameters(X_train, y_train):
    params = {
        "max_depth": [5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", "unbalanced"], 
        "splitter": ["best", "random"]
    }
    clf = DecisionTreeClassifier(random_state=0)
    random_search = RandomizedSearchCV(
        clf,
        param_distributions=params,
        n_iter=40,
        cv=5,
        n_jobs=-1,
        random_state=0
    )
    random_search.fit(X_train, y_train)
    print(" -> Melhores parâmetros encontrados:", random_search.best_params_)
    return random_search.best_estimator_

def print_tree_structure(clf, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, rounded=True, feature_names=feature_names, class_names=True)
    plt.title("Árvore de Decisão - Visualização")
    plt.show()

def run_experiment_for_ratio(df, non_medal_ratio):
    print(f"\n*** Proporção: {non_medal_ratio*100:.0f}% não medalhistas / {(1-non_medal_ratio)*100:.0f}% medalhistas ***")
    df_subset = create_subset_by_ratio(df, non_medal_ratio)
    X, y = preprocess_data(df_subset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    best_model = tune_hyperparameters(X_train, y_train)
    #print("Visualizando a árvore de decisão:")
    #print_tree_structure(best_model, list(X.columns))
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Acurácia:", acc)
    print("Relatório de Classificação:")
    print(report)

def run_experiment_full(df):
    print("\n*** Usando todo o conjunto de dados ***")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    best_model = tune_hyperparameters(X_train, y_train)
    #print("Visualizando a árvore de decisão:")
    #print_tree_structure(best_model, list(X.columns))
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Acurácia:", acc)
    print("Relatório de Classificação:")
    print(report)

def main():
    df = load_data()
    run_experiment_full(df)
    ratios = [0.40, 0.45, 0.50, 0.55, 0.60]
    for ratio in ratios:
        run_experiment_for_ratio(df, ratio)

if __name__ == "__main__":
    main()