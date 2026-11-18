import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

print("======================================")
print("   MACHINE LEARNING MODEL TRAINING")
print("======================================")


# STEP 1: Load similarity dataset

print("\n1. Loading dataset...")

df = pd.read_csv("dataset/similarity_results.csv")

df["question1"] = df["question1"].fillna("").astype(str)
df["question2"] = df["question2"].fillna("").astype(str)
df["similarity_score"] = df["similarity_score"].fillna(0)

print("Dataset loaded:", df.shape)


# STEP 2: Feature extraction

print("\n2. Extracting features...")


def common_word_ratio(q1, q2):

    words1 = set(q1.lower().split())
    words2 = set(q2.lower().split())

    if not words1 or not words2:
        return 0

    common = words1.intersection(words2)

    return len(common) / max(len(words1), len(words2))


df["q1_length"] = df["question1"].apply(
    lambda x: len(x.split())
)

df["q2_length"] = df["question2"].apply(
    lambda x: len(x.split())
)

df["common_word_ratio"] = df.apply(
    lambda row: common_word_ratio(
        row["question1"],
        row["question2"]
    ),
    axis=1
)


# STEP 3: Prepare X and Y

print("\n3. Preparing training data...")

X = df[
    [
        "similarity_score",
        "q1_length",
        "q2_length",
        "common_word_ratio"
    ]
]

y = df["is_duplicate"].astype(int)


# STEP 4: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 5: Logistic Regression

print("\n4. Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("Logistic Regression completed!")


# STEP 6: Random Forest

print("\n5. Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("Random Forest completed!")


# STEP 7: Evaluation function

def evaluate_model(name, actual, predicted):

    print("\n================================")
    print(name)
    print("================================")

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        actual,
        predicted,
        zero_division=0
    ))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# STEP 8: Evaluate both models

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)


# STEP 9: Save model comparison

comparison = pd.DataFrame([
    lr_results,
    rf_results
])

print("\nMODEL COMPARISON:")
print(comparison)

comparison.to_csv(
    "model/model_comparison.csv",
    index=False
)


# STEP 10: Confusion Matrix

print("\nCreating confusion matrix...")

cm = confusion_matrix(y_test, lr_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Duplicate", "Duplicate"],
    yticklabels=["Not Duplicate", "Duplicate"]
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix.png"
)

plt.show()


# STEP 11: Save trained models

print("\nSaving trained models...")

joblib.dump(
    lr_model,
    "model/logistic_regression.pkl"
)

joblib.dump(
    rf_model,
    "model/random_forest.pkl"
)

joblib.dump(
    list(X.columns),
    "model/feature_columns.pkl"
)

print("\n======================================")
print(" MACHINE LEARNING TRAINING COMPLETED!")
print("======================================")

print("\nGenerated files:")
print("1. model/logistic_regression.pkl")
print("2. model/random_forest.pkl")
print("3. model/feature_columns.pkl")
print("4. model/model_comparison.csv")
print("5. screenshots/confusion_matrix.png")