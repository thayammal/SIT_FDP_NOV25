import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
df = pd.read_csv("placement_details.csv")
X = df.drop("Placement", axis=1)
y = df["Placement"]

# 2. Identify numeric & categorical columns
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 3. Build preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 4. Define base estimator (a shallow decision tree)  
base_dt = DecisionTreeClassifier(random_state=42)

# 5. Build full pipeline with AdaBoost
ada_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ada', AdaBoostClassifier(
        estimator=base_dt,       # set the base estimator
        random_state=42
    ))
])

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Define parameter grid for tuning both AdaBoost and base estimator
param_grid = {
    'ada__n_estimators': [50, 100, 150],
    'ada__learning_rate': [0.01, 0.1, 1.0],
    'ada__estimator__max_depth': [1, 2, 3],          # tuning base estimator
    'ada__estimator__min_samples_split': [2, 5, 10], # tuning base estimator
}

# 8. Setup GridSearchCV
grid_search = GridSearchCV(
    ada_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# 9. Fit grid search
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-val accuracy:", grid_search.best_score_)

# 10. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 11. Save the best model
joblib.dump(best_model, "ada_boost_best_model.pkl")
print("Saved model to ada_boost_best_model.pkl")
