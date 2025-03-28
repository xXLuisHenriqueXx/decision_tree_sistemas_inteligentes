import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path="athlete_events_cleaned.csv"):
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    X = data.drop("Medal_won", axis=1)
    y = data["Medal_won"]
    return X, y

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

def search_params(X_train, y_train):
    params = {
        "max_depth": [5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", "unbalanced"],
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

def show_tree(model, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, rounded=True, feature_names=feature_names, class_names=True)
    plt.title("Decision Tree View")
    plt.show()

def run_ratio_experiment(data, non_medal_ratio):
    print("\n*** Running experiment with:")
    print("    {}% of athletes WITHOUT medals".format(int(non_medal_ratio * 100)))
    print("   {}% of athletes WITH medals ***".format(int((1 - non_medal_ratio) * 100)))
    subset = get_subset(data, non_medal_ratio)
    X, y = prepare_data(subset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    best_model = search_params(X_train, y_train)

    # show_tree(best_model, list(X.columns))
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    rep = classification_report(y_test, predictions)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(rep)

def run_full_experiment(data):
    print("\n*** Using the full dataset ***")
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    best_model = search_params(X_train, y_train)

    # show_tree(best_model, list(X.columns))
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