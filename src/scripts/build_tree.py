import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
import json
import subprocess

warnings.filterwarnings("ignore")

def load_data(file_path="athlete_events_cleaned.csv"):
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    X = data.drop("Medal_won", axis=1)
    y = data["Medal_won"]
    return X, y

def get_subset(data, non_medal_ratio):
    # Seleciona todas as ocorrências com medalha
    with_medal = data[data["Medal_won"] == 1]
    without_medal = data[data["Medal_won"] == 0]
    num_medals = len(with_medal)

    desired_without = int((non_medal_ratio / (1 - non_medal_ratio)) * num_medals)
    if desired_without > len(without_medal):
        desired_without = len(without_medal)
    sample_without = without_medal.sample(n=desired_without, random_state=0)
    subset = pd.concat([with_medal, sample_without]).sample(frac=1, random_state=0)
    return subset

def search_params(X_train, y_train):
    params = {
        "max_depth": [5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
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
    print(" -> Best parameters found (Random Search):", search.best_params_)
    return search.best_estimator_

def search_all_params(X_train, y_train):
    params = {
        "max_depth": list(range(1, 31)),
        "min_samples_split": list(range(2, 21)),
        "min_samples_leaf": list(range(1, 11)),
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
        "splitter": ["best", "random"]
    }
    model = DecisionTreeClassifier(random_state=0)
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(" -> Best parameters found (Grid Search):", grid.best_params_)
    return grid.best_estimator_

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
    # Ajusta o DOT para diminuir o fontsize dos nós
    with open(dot_file, "r") as f:
        lines = f.readlines()
    new_lines = []
    for idx, line in enumerate(lines):
        new_lines.append(line)
        if idx == 0 and "digraph" in line:
            new_lines.append('    node [fontsize=6, margin=0.05];\n')
    with open(dot_file, "w") as f:
        f.writelines(new_lines)
    # Converte o DOT para SVG (formato vetorial)
    subprocess.run(["dot", "-Tsvg", dot_file, "-o", output_svg])
    print("SVG gerado:", output_svg)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc

def run_subset_experiment(data, non_medal_ratio, search_func):
    subset = get_subset(data, non_medal_ratio)
    X_train, y_train = prepare_data(subset)
    # O conjunto de teste é composto por todas as ocorrências que não foram selecionadas no subset
    test_data = data.drop(subset.index)
    if test_data.shape[0] == 0:
        # Se não houver dados de teste, realiza um split normal
        X_full, y_full = prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=0)
    else:
        X_test, y_test = prepare_data(test_data)
    model = search_func(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    return model, acc

def run_full_experiment(data, search_func):
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = search_func(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    return model, acc

def run_tests():
    print("\n--- Iniciando testes ---")
    data = load_data()
    if data.empty:
        print("Erro: dataset vazio!")
        return

    # Teste da função prepare_data
    X, y = prepare_data(data)
    assert "Medal_won" not in X.columns, "prepare_data não removeu a coluna 'Medal_won'"
    print("Teste prepare_data OK.")

    # Teste da função get_subset
    subset = get_subset(data, 0.5)
    if subset.empty:
        print("Erro: subset vazio!")
        return
    print("Teste get_subset OK.")

    # Teste dos métodos de busca com um pequeno split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model_rand = search_params(X_train, y_train)
    model_grid = search_all_params(X_train, y_train)
    acc_rand = evaluate_model(model_rand, X_test, y_test)
    acc_grid = evaluate_model(model_grid, X_test, y_test)
    print(f"Teste busca: Random Search Accuracy = {acc_rand:.3f}, Grid Search Accuracy = {acc_grid:.3f}")

    print("--- Testes concluídos ---\n")

def main():
    data = load_data()
    results = []
    
    # Experimentos com dataset completo
    model_full_rand, acc_full_rand = run_full_experiment(data, search_params)
    results.append({"experiment": "full_random", "accuracy": acc_full_rand, "hyperparameters": model_full_rand.get_params()})
    
    model_full_grid, acc_full_grid = run_full_experiment(data, search_all_params)
    results.append({"experiment": "full_grid", "accuracy": acc_full_grid, "hyperparameters": model_full_grid.get_params()})
    
    # Experimentos com subsets para diferentes ratios
    for ratio in [0.40, 0.45, 0.50, 0.55, 0.60]:
        model_sub_rand, acc_sub_rand = run_subset_experiment(data, ratio, search_params)
        results.append({
            "experiment": f"subset_random_{int(ratio*100)}",
            "accuracy": acc_sub_rand,
            "hyperparameters": model_sub_rand.get_params()
        })
        model_sub_grid, acc_sub_grid = run_subset_experiment(data, ratio, search_all_params)
        results.append({
            "experiment": f"subset_grid_{int(ratio*100)}",
            "accuracy": acc_sub_grid,
            "hyperparameters": model_sub_grid.get_params()
        })
    
    # Determina o melhor experimento baseado na acurácia
    best = max(results, key=lambda r: r["accuracy"])
    print("\n=== Melhor Experimento ===")
    print("Experimento:", best["experiment"])
    print("Acurácia:", best["accuracy"])
    print("Hiperparâmetros:", best["hyperparameters"])
    
    # Salva os melhores hiperparâmetros em um arquivo JSON
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best, f, indent=4)
    print("Hiperparâmetros salvos em best_hyperparameters.json")
    
    # Opcional: Mostrar a árvore do melhor modelo se o dataset usado for o completo
    X, y = prepare_data(data)
    if best["experiment"] in ["full_random", "full_grid"]:
        show_tree(best_model := (model_full_rand if best["experiment"]=="full_random" else model_full_grid),
                  list(X.columns), "decision_tree_full.svg")

if __name__ == "__main__":
    run_tests()
    main()