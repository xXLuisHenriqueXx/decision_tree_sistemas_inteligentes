import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath="athlete_events_cleaned.csv"):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Separa as features (exceto a coluna alvo) e o target
    X = df.drop("Medal_won", axis=1)
    y = df["Medal_won"]
    return X, y

def tune_hyperparameters(X_train, y_train, cv_folds=5, n_iter=60):
    # Expande o espaço de hiperparâmetros para uma busca mais abrangente
    param_dist = {
        "max_depth": [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200],
        "min_samples_split": [2, 3, 5, 10, 20, 30, 50, 75, 100],
        "min_samples_leaf": [1, 2, 5, 10, 20, 30, 50],
        "criterion": ["gini", "entropy"]
    }
    clf = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    print(" -> Melhores parâmetros encontrados:", random_search.best_params_)
    return random_search.best_estimator_

def create_subset_by_ratio(df, non_medal_ratio):
    """
    Cria um subconjunto do dataset com a proporção desejada.
    non_medal_ratio: proporção desejada para não medalhistas (por exemplo, 0.4 para 40%).
    Usa todos os medalhistas e amostragem aleatória dos não medalhistas para atingir a razão:
        non_medal : medal = non_medal_ratio : (1 - non_medal_ratio)
    """
    df_medal = df[df["Medal_won"] == 1]
    df_non_medal = df[df["Medal_won"] == 0]
    n_medal = len(df_medal)
    target_non_medal = int((non_medal_ratio / (1 - non_medal_ratio)) * n_medal)
    if target_non_medal > len(df_non_medal):
        target_non_medal = len(df_non_medal)
    df_non_sampled = df_non_medal.sample(n=target_non_medal, random_state=42)
    df_subset = pd.concat([df_non_sampled, df_medal]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_subset

def run_experiment_for_ratio(df, non_medal_ratio):
    print("\n*** Proporção: {:.0f}% não medalhistas / {:.0f}% medalhistas ***".format(
        non_medal_ratio*100, (1-non_medal_ratio)*100))
    df_subset = create_subset_by_ratio(df, non_medal_ratio)
    X, y = preprocess_data(df_subset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Buscamos 60 combinações de hiperparâmetros no espaço expandido
    best_clf = tune_hyperparameters(X_train, y_train, cv_folds=5, n_iter=60)
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Acurácia:", acc)
    print("Relatório de Classificação:")
    print(report)
    # Descomente as linhas abaixo para visualizar as regras da árvore
    # tree_rules = export_text(best_clf, feature_names=list(X.columns))
    # print("Regras da Árvore:")
    # print(tree_rules)

def main():
    df = load_data()
    # Focar nos subconjuntos promissores: testamos várias proporções entre 40% e 60%
    ratios = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
    for ratio in ratios:
        run_experiment_for_ratio(df, ratio)

if __name__ == "__main__":
    main()