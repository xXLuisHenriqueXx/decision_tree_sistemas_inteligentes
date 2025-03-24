import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath="breast_cancer_cleaned.csv"):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    return X, y

def tune_hyperparameters(X_train, y_train):
    param_dist = {
        "max_depth": [5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", "unbalanced"],  # Balancing classes
        "splitter": ["best", "random"]
    }
    
    clf = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=40, cv=5, n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    print("Best Parameters:", random_search.best_params_)
    
    return random_search.best_estimator_

def main():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    best_clf = tune_hyperparameters(X_train, y_train)
    
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", acc)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
