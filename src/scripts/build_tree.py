import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import subprocess
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path="athlete_events_cleaned.csv"):
    data = pd.read_csv(file_path)
    return data

# Prepara os dados para o treinamento do modelo
# separando as features (x) e o alvo (y)
def prepare_data(data):
    X = data.drop("Medal_won", axis=1)
    y = data["Medal_won"]
    return X, y

# Dado um dataframe, retorna um subconjunto com uma proporção específica de atletas com e sem medalhas
def get_subset(data, non_medal_ratio):
    with_medal = data[data["Medal_won"] == 1]
    without_medal = data[data["Medal_won"] == 0]
    num_medals = len(with_medal)

    desired_without = int((non_medal_ratio / (1 - non_medal_ratio)) * num_medals)
    if desired_without > len(without_medal):
        desired_without = len(without_medal)
    sample_without = without_medal.sample(n=desired_without, random_state=0)
    subset = pd.concat([with_medal, sample_without]).sample(frac=1, random_state=0).reset_index(drop=True)
    return subset

# Dado um dataframe, retorna um modelo de árvore de decisão treinado
# com os melhores parâmetros encontrados por busca aleatória
def search_params(X_train, y_train):
    params = {
        "max_depth": range(1, 31),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 10),
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", "unbalanced", None],
        "splitter": ["best", "random"]
    }
    model = DecisionTreeClassifier(random_state=0)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=40,
        cv=5,
        n_jobs=-1,
        random_state=0
    )
    search.fit(X_train, y_train)
    print(" -> Best parameters found:", search.best_params_)
    return search.best_estimator_


# Função para visualização da árvore em formato SVG
def show_tree(model, feature_names, output_svg="decision_tree_full.svg"):

    dot_file = "tree.dot"
    export_graphviz(
        model,
        out_file=dot_file,
        feature_names=feature_names,
        class_names=["Sem medalha", "Com medalha"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    with open(dot_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for idx, line in enumerate(lines):
        new_lines.append(line)
        if idx == 0 and "digraph" in line:
            new_lines.append('    node [fontsize=6, margin=0.05];\n')
    with open(dot_file, "w") as f:
        f.writelines(new_lines)
    
    subprocess.run(["dot", "-Tsvg", dot_file, "-o", output_svg])
    os.remove(dot_file)
    print("SVG gerado:", output_svg)

# Roda o treinamento e teste do modelo com uma proporção específica de atletas com e sem medalhas
# e gera o gráfico da árvore de decisão
def run_ratio_experiment(data, non_medal_ratio):
    print("\n*** Running experiment with:")
    print("   {}% of athletes WITHOUT medals".format(int(non_medal_ratio * 100)))
    print("   {}% of athletes WITH medals ***".format(int((1 - non_medal_ratio) * 100)))
    subset = get_subset(data, non_medal_ratio)
    X, y = prepare_data(subset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    best_model = search_params(X_train, y_train)

    show_tree(best_model, list(X.columns), "decision_tree_{}_{}.svg".format(int(non_medal_ratio * 100), int((1 - non_medal_ratio) * 100)))
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    rep = classification_report(y_test, predictions)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(rep)

# Roda o treinamento e teste do modelo com o dataset completo
def run_full_experiment(data):
    print("\n** Using the full dataset **")
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    best_model = search_params(X_train, y_train)

    show_tree(best_model, list(X.columns), "decision_tree_full.svg")
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    rep = classification_report(y_test, predictions)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(rep)

def main():
    data = load_data()
    run_full_experiment(data)
    ratios = [0.40, 0.45, 0.50, 0.55, 0.60]
    for rate in ratios:
         run_ratio_experiment(data, rate)

if __name__ == "__main__":
    main()


# DÚVIDAS:
# - A precisão é suficiente?
# - O documento é suficiente?
# - As árvores geradas com proporções estão boas? São meio grandes
# - O dataset escolhido é bom? Devemos troca-los ou aumentar a abrangência de esportes?
# - A definição de parâmetros por busca randomizada é suficiente?
