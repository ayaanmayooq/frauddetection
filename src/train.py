from src.data_prep import prepare_data
from src.models import logistic_regression_model, decision_tree_model, random_forest_model, lightgbm_model, neural_network_model
from src.loss_functions import focal_loss
from src.evaluation import evaluate_model

def train_sklearn_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

def train_nn_model(X_train, y_train, X_test, y_test):
    model = neural_network_model(X_train.shape[1])
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['AUC'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    evaluate_model(y_test, y_pred)