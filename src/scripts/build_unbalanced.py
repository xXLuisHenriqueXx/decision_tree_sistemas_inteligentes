import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath="athlete_events_cleaned.csv"):
    # Lê o dataset completo (desbalanceado)
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Separa as features (exceto a coluna alvo "Medal_won") e o target.
    X = df.drop("Medal_won", axis=1)
    y = df["Medal_won"]
    return X, y

def tune_hyperparameters(X_train, y_train, cv_folds=5, n_iter=100):
    # Espaço de busca expansivo, incluindo class_weight e splitter para ajudar com o desequilíbrio.
    param_dist = {
        "max_depth": [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200],
        "min_samples_split": [2, 3, 5, 10, 20, 30, 50, 75, 100],
        "min_samples_leaf": [1, 2, 5, 10, 20, 30, 50],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced"],
        "splitter": ["best", "random"]
    }
    clf = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_iter,          # 100 iterações para uma busca abrangente.
        cv=cv_folds,            # Validação cruzada com 5 folds.
        n_jobs=-1,              # Usa todos os núcleos disponíveis.
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    print(" -> Melhores parâmetros encontrados:", random_search.best_params_)
    return random_search.best_estimator_

def main():
    # Carrega o dataset completo/desbalanceado.
    df = load_data()
    print("Dataset original:", df.shape)
    
    # Pré-processamento: separa features e target.
    X, y = preprocess_data(df)
    
    # Divisão estratificada para manter a distribuição original das classes.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Busca extensiva de hiperparâmetros.
    best_clf = tune_hyperparameters(X_train, y_train, cv_folds=5, n_iter=100)
    
    # Predição e avaliação.
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Acurácia:", acc)
    print("Relatório de Classificação:")
    print(report)
    
    # Opcional: para visualizar as regras da árvore, descomente as linhas abaixo.
    # tree_rules = export_text(best_clf, feature_names=list(X.columns))
    # print("Regras da Árvore:")
    # print(tree_rules)

if __name__ == "__main__":
    main()
