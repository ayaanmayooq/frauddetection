from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras import layers

def logistic_regression_model():
    return LogisticRegression()

def decision_tree_model():
    return DecisionTreeClassifier()

def random_forest_model():
    return RandomForestClassifier()

def lightgbm_model():
    return LGBMClassifier()

def neural_network_model(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model