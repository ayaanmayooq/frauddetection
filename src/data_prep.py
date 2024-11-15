import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATHS = {
    'creditcard': 'data/creditcard/creditcard.csv',
}

def load_data(dataset_name):
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(DATASET_PATHS.keys())}")

    if dataset_name == 'creditcard':
        data = pd.read_csv(DATASET_PATHS[dataset_name])
        X = data.drop(['Class'], axis=1)
        y = data['Class']
        return X, y

    elif dataset_name == 'other':
        pass


def preprocess_data(X_train, X_test, dataset_name):
    if dataset_name == 'creditcard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    elif dataset_name == 'other':
        pass


def prepare_data(dataset_name):
    X, y = load_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = preprocess_data(X_train, X_test, dataset_name)
    return X_train, X_test, y_train, y_test