import os
import re
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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
print(" ADVANCED TEXT SIMILARITY DETECTION")
print("======================================")


# STEP 1: Load dataset

print("\n1. Loading dataset...")

df = pd.read_csv("dataset/cleaned_questions.csv")

df["question1"] = df["question1"].fillna("").astype(str)
df["question2"] = df["question2"].fillna("").astype(str)

df = df[
    (df["question1"].str.strip() != "") &
    (df["question2"].str.strip() != "")
].copy()

df["is_duplicate"] = df["is_duplicate"].astype(int)

print("Dataset loaded:", df.shape)


# STEP 2: Select sample

print("\n2. Selecting dataset sample...")

df = df.sample(
    n=min(100000, len(df)),
    random_state=42
).reset_index(drop=True)

print("Sample size:", len(df))


# STEP 3: Split dataset

print("\n3. Splitting dataset...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["is_duplicate"]
)

print("Training:", len(train_df))
print("Testing:", len(test_df))


# STEP 4: Create TF-IDF vectorizer

print("\n4. Creating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode"
)

training_questions = pd.concat([
    train_df["question1"],
    train_df["question2"]
])

vectorizer.fit(training_questions)

print("TF-IDF training completed!")


# STEP 5: Convert questions into vectors

print("\n5. Creating question vectors...")

q1_train = vectorizer.transform(train_df["question1"])
q2_train = vectorizer.transform(train_df["question2"])

q1_test = vectorizer.transform(test_df["question1"])
q2_test = vectorizer.transform(test_df["question2"])

print("Question vectors created!")


# STEP 6: Create numerical features

def create_numeric_features(q1, q2, v1, v2):

    features = []

    for text1, text2, vec1, vec2 in zip(q1, q2, v1, v2):

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        common = words1.intersection(words2)

        union = words1.union(words2)

        overlap = (
            len(common) / max(len(words1), len(words2), 1)
        )

        jaccard = (
            len(common) / max(len(union), 1)
        )

        length1 = len(words1)
        length2 = len(words2)

        length_difference = abs(length1 - length2)

        length_ratio = (
            min(length1, length2) /
            max(length1, length2, 1)
        )

        cosine = float(vec1.multiply(vec2).sum())

        features.append([
            cosine,
            overlap,
            jaccard,
            np.log1p(length1),
            np.log1p(length2),
            length_difference,
            length_ratio
        ])

    return csr_matrix(np.array(features, dtype=np.float32))


print("\n6. Extracting additional features...")

numeric_train = create_numeric_features(
    train_df["question1"].tolist(),
    train_df["question2"].tolist(),
    q1_train,
    q2_train
)

numeric_test = create_numeric_features(
    test_df["question1"].tolist(),
    test_df["question2"].tolist(),
    q1_test,
    q2_test
)


# STEP 7: Create pairwise TF-IDF features

print("\n7. Creating pairwise features...")

def create_pair_features(v1, v2, numeric):

    difference = abs(v1 - v2)

    product = v1.multiply(v2)

    return hstack([
        difference,
        product,
        numeric
    ], format="csr")


X_train = create_pair_features(
    q1_train,
    q2_train,
    numeric_train
)

X_test = create_pair_features(
    q1_test,
    q2_test,
    numeric_test
)

y_train = train_df["is_duplicate"]
y_test = test_df["is_duplicate"]

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)


# STEP 8: Train Logistic Regression

print("\n8. Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    solver="liblinear"
)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("Logistic Regression completed!")


# STEP 9: Train Linear SVM

print("\n9. Training Linear SVM...")

svm_model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

print("Linear SVM completed!")


# STEP 10: Evaluate models

def evaluate_model(name, actual, predicted):

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    print("\n================================")
    print(name)
    print("================================")

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


print("\n10. Evaluating models...")

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)

svm_results = evaluate_model(
    "Linear SVM",
    y_test,
    svm_predictions
)


# STEP 11: Compare models

comparison = pd.DataFrame([
    lr_results,
    svm_results
])

print("\nMODEL COMPARISON")

print(comparison)

os.makedirs("model", exist_ok=True)

comparison.to_csv(
    "model/model_comparison.csv",
    index=False
)


# STEP 12: Select best model

best_result = comparison.loc[
    comparison["F1 Score"].idxmax()
]

best_model_name = best_result["Model"]

if best_model_name == "Linear SVM":
    best_model = svm_model
    best_predictions = svm_predictions
else:
    best_model = lr_model
    best_predictions = lr_predictions

print("\nBest Model:", best_model_name)


# STEP 13: Save models

print("\n11. Saving models...")

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)

joblib.dump(
    lr_model,
    "model/logistic_regression.pkl"
)

joblib.dump(
    svm_model,
    "model/linear_svm.pkl"
)

joblib.dump(
    best_model,
    "model/best_model.pkl"
)

joblib.dump(
    best_model_name,
    "model/best_model_name.pkl"
)

print("Models saved successfully!")


# STEP 14: Confusion matrix

print("\n12. Creating confusion matrix...")

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Duplicate", "Duplicate"],
    yticklabels=["Not Duplicate", "Duplicate"]
)

plt.title("Advanced Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

os.makedirs("screenshots", exist_ok=True)

plt.savefig(
    "screenshots/advanced_confusion_matrix.png"
)

plt.show()


# STEP 15: Final output

print("\n======================================")
print(" ADVANCED MODEL TRAINING COMPLETED!")
print("======================================")

print("Selected Model:", best_model_name)

print("\nGenerated Files:")

print("1. model/tfidf_vectorizer.pkl")
print("2. model/logistic_regression.pkl")
print("3. model/linear_svm.pkl")
print("4. model/best_model.pkl")
print("5. model/best_model_name.pkl")
print("6. model/model_comparison.csv")

print("7. screenshots/advanced_confusion_matrix.png")

print("\nSUCCESS!")