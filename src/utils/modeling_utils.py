from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Function to train and evaluate the SVM model
def svm(X_train, y_train, params=None):
    print("Training SVM...")
    svm_model = SVC(kernel='linear', random_state=42)
    
    if params:
        grid_search = GridSearchCV(svm_model, params, cv=5, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    svm_model.fit(X_train, y_train)
    return svm_model

# Function to train and evaluate the KNN model
def knn(X_train, y_train, params=None):
    print("Training KNN...")
    knn_model = KNeighborsClassifier()
    
    if params:
        grid_search = GridSearchCV(knn_model, params, cv=5, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    knn_model.fit(X_train, y_train)
    return knn_model

# Function to train and evaluate the Random Forest model
def random_forest(X_train, y_train, params=None):
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42)
    
    if params:
        grid_search = GridSearchCV(rf_model, params, cv=5, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to train and evaluate the LightGBM model
def lgbm(X_train, y_train, params=None):
    print("Training LGBM...")
    lgbm_model = LGBMClassifier(random_state=42)
    
    if params:
        grid_search = GridSearchCV(lgbm_model, params, cv=5, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    lgbm_model.fit(X_train, y_train)
    return lgbm_model

# Function to train and evaluate the XGBoost model
def xgboost(X_train, y_train, params=None):
    print("Training XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    if params:
        grid_search = GridSearchCV(xgb_model, params, cv=5, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    xgb_model.fit(X_train, y_train)
    return xgb_model

# class CustomGridSearchCV(GridSearchCV):
#     def _run_search(self, parameter_candidates):
#         for params in parameter_candidates:
#             self.estimator.set_params(**params)
#             scores = cross_val_score(self.estimator, self.X, self.y, cv=self.cv, scoring='accuracy')
#             print(f"Parameters: {params}, Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
#             self._store_results(params, scores)