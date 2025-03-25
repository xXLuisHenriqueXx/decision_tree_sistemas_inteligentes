import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def load_data(filepath="breast_cancer_cleaned.csv"):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    return X, y

def get_fixed_classifier():
    clf = DecisionTreeClassifier(
        splitter="random",
        min_samples_split=2,
        min_samples_leaf=5,
        max_depth=30,
        criterion="entropy",
        class_weight="balanced",
        random_state=0
    )
    return clf

def print_tree_structure(clf, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, rounded=True, feature_names=feature_names, class_names=True)
    plt.title("Árvore de Decisão - Visualização")
    plt.show()
    tree_text = export_text(clf, feature_names=list(feature_names))
    print("Estrutura da Árvore (texto):")
    print(tree_text)

def main():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = get_fixed_classifier()
    clf.fit(X_train, y_train)
    print("Parâmetros fixos:", clf.get_params())
    print_tree_structure(clf, X.columns)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Acurácia:", acc)
    print("Relatório de Classificação:\n", report)

if __name__ == "__main__":
    main()