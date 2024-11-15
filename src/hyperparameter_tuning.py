from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best AUC-ROC:", grid_search.best_score_)
    return grid_search.best_estimator_