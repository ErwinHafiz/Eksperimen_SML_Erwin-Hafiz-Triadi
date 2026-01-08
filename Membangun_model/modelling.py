import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub


# INIT DAGSHUB + MLFLOW
dagshub.init(
    repo_owner="erwinhafizzxr",
    repo_name="titanic-mlflow-erwin-hafiz-triadi",
    mlflow=True
)


# LOAD DATA
df = pd.read_csv("dataset_preprocessing/titanic_clean.csv")

X = df.drop("2urvived", axis=1)
y = df["2urvived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()
# MLFLOW EXPERIMENT
# mlflow.set_experiment("Titanic-Baseline")

# with mlflow.start_run():
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# preds = model.predict(X_test)
# acc = accuracy_score(y_test, preds)
# LOGGING
# mlflow.log_metric("accuracy", acc)
# mlflow.sklearn.log_model(model, "model")