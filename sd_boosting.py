import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load data
data = pd.read_csv("placement_details.csv")

# 2. Prepare features & target
X = data.drop("Placement", axis=1)
y = data["Placement"]

# 3. Identify numeric & categorical columns
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

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

# 5. Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Define a boosting model (choose one)
boost_model = Pipeline([
    ("preprocessor", preprocessor),
    ("gbc", GradientBoostingClassifier( #  DecisionTreeRegressor with max_depth=3.
         n_estimators=100,
         learning_rate=0.1,
         max_depth=3,
         random_state=42
    ))
])

# Alternate: AdaBoost
# boost_model = Pipeline([
#     ("preprocessor", preprocessor),
#     ("ada", AdaBoostClassifier(
#         n_estimators=50,
#         learning_rate=1.0,
#         random_state=42
#     ))
# ])

# 7. Fit model
boost_model.fit(X_train, y_train)

# 8. Predict & evaluate
y_pred = boost_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Save the model
joblib.dump(boost_model, "placement_boosting_model.pkl")
print("Model saved as placement_boosting_model.pkl")
