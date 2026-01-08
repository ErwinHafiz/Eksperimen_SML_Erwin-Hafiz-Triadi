import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import dagshub
import os

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


# HYPERPARAMETER TUNING

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy"
)

mlflow.set_experiment("Titanic-Tuning")

with mlflow.start_run():
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    
    # MANUAL LOGGING
    
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)

    
    # ARTIFACT 1: CONFUSION MATRIX
    
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    
    # ARTIFACT 2: CLASSIFICATION REPORT
    
    report = classification_report(y_test, preds)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    
    # SAVE MODEL
    
    mlflow.sklearn.log_model(best_model, "model")
