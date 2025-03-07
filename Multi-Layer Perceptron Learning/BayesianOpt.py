from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestClassifier()

# Define the hyperparameter search space
param_space = {
    'n_estimators': (10, 200),          # Number of trees in the forest
    'max_depth': (3, 50),               # Maximum depth of trees
    'min_samples_split': (2, 10),       # Minimum samples required to split a node
    'min_samples_leaf': (1, 5)          # Minimum samples required in a leaf node
}

# Bayesian Optimization with 10 iterations and 5-fold cross-validation
opt = BayesSearchCV(rf, param_space, n_iter=10, cv=5, random_state=42, n_jobs=-1)
opt.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", opt.best_params_)
print("Best Score:", opt.best_score_)
