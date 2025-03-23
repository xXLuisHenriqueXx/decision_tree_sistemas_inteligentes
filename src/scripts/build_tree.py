import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath="athlete_events_cleaned.csv"):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Separa as features (todas, exceto a coluna alvo) e o target
    X = df.drop("Medal_won", axis=1)
    y = df["Medal_won"]
    return X, y

def tune_hyperparameters(X_train, y_train):
    # Expande a busca de hiperparâmetros com uma distribuição maior
    param_dist = {
        "max_depth": [5, 7, 10, 15, 20, 25, 30],
        "min_samples_split": [5, 10, 20, 50],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion": ["gini", "entropy"]
    }
    clf = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=40,      # Número de combinações a serem testadas
        cv=5,           # Utiliza 5 folds para uma avaliação mais robusta
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    print("Melhores parâmetros encontrados:", random_search.best_params_)
    return random_search.best_estimator_

def main():
    # Carrega os dados
    df = load_data()
    
    # Pré-processamento
    X, y = preprocess_data(df)
    
    # Divisão dos dados: 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Busca dos melhores hiperparâmetros com a nova configuração
    best_clf = tune_hyperparameters(X_train, y_train)
    
    # Predição e avaliação
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Acurácia:", acc)
    print("Relatório de Classificação:")
    print(report)
    
    # Descomente se desejar ver as regras da árvore
    # tree_rules = export_text(best_clf, feature_names=list(X.columns))
    # print("Regras da Árvore:")
    # print(tree_rules)

if __name__ == "__main__":
    main()