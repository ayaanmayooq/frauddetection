# main.py
from src.data_prep import prepare_data
from src.train import train_sklearn_model, train_nn_model
from src.models import logistic_regression_model, decision_tree_model, random_forest_model, lightgbm_model


def main():
    dataset_name = 'creditcard'

    X_train, X_test, y_train, y_test = prepare_data(dataset_name)

    print(f"\n--- Training models for {dataset_name} dataset ---\n")

    print("Logistic Regression:")
    train_sklearn_model(logistic_regression_model(), X_train, y_train, X_test, y_test)

    print("\nDecision Tree:")
    train_sklearn_model(decision_tree_model(), X_train, y_train, X_test, y_test)

    print("\nRandom Forest:")
    train_sklearn_model(random_forest_model(), X_train, y_train, X_test, y_test)

    print("\nLightGBM:")
    train_sklearn_model(lightgbm_model(), X_train, y_train, X_test, y_test)

    print("\nNeural Network with Focal Loss:")
    train_nn_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
