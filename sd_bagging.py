import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load data
data = pd.read_csv("placement_details.csv")

# 2. Features / target
X = data.drop("Placement", axis=1)
y = data["Placement"]

# 3. Identify numeric & categorical columns
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

# 4. Preprocessing pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 5. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Define base model + bagging ensemble
base_model = LogisticRegression(random_state=42, max_iter=1000)

bagging_model = Pipeline([
    ("preprocessor", preprocessor),
    ("bagging", BaggingClassifier(
         estimator=base_model,
         n_estimators=50,
         max_samples=0.8,
         max_features=0.8,
         bootstrap=True,
         bootstrap_features=False,
         random_state=42,
         n_jobs=-1
    ))
])

# 7. Fit model
bagging_model.fit(X_train, y_train)

# 8. Evaluate
y_pred = bagging_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# 9. Save the model
joblib.dump(bagging_model, "placement_bagging_model.pkl")
print("Model saved as placement_bagging_model.pkl")
