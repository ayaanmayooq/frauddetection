from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred):
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("AUC-ROC:", roc_auc_score(y_true, y_pred))