import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

print("===================================")
print(" TEXT SIMILARITY FEATURE ENGINEERING")
print("===================================")

# STEP 1: Load cleaned dataset

print("\n1. Loading dataset...")

df = pd.read_csv("dataset/cleaned_questions.csv")

print("Dataset loaded successfully!")
print("Original Dataset Shape:", df.shape)


# STEP 2: Handle missing values

print("\n2. Handling missing values...")

df["question1"] = df["question1"].fillna("").astype(str)
df["question2"] = df["question2"].fillna("").astype(str)

# Remove rows containing empty questions

df = df[
    (df["question1"].str.strip() != "") &
    (df["question2"].str.strip() != "")
].copy()

print("Missing values handled!")
print("Dataset Shape After Cleaning:", df.shape)


# STEP 3: Select sample

print("\n3. Selecting dataset sample...")

df = df.sample(
    n=min(100000, len(df)),
    random_state=42
).reset_index(drop=True)

print("Sample selected:", df.shape)


# STEP 4: Split dataset into training and testing

print("\n4. Splitting dataset...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["is_duplicate"]
)

print("Training Records:", len(train_df))
print("Testing Records:", len(test_df))


# STEP 5: Create TF-IDF Vectorizer

print("\n5. Creating TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


# STEP 6: Train TF-IDF Vectorizer

print("\n6. Fitting TF-IDF...")

train_questions = pd.concat(
    [
        train_df["question1"],
        train_df["question2"]
    ],
    ignore_index=True
)

vectorizer.fit(train_questions)

print("TF-IDF fitting completed!")


# STEP 7: Convert questions into numerical vectors

print("\n7. Converting questions into vectors...")

q1_vectors = vectorizer.transform(
    test_df["question1"]
)

q2_vectors = vectorizer.transform(
    test_df["question2"]
)

print("Question vectors created successfully!")


# STEP 8: Calculate Cosine Similarity

print("\n8. Calculating Cosine Similarity...")

scores = cosine_similarity(
    q1_vectors,
    q2_vectors,
    dense_output=False
).diagonal()

print("\nFirst 10 Similarity Scores:")

for score in scores[:10]:
    print(round(float(score), 4))


# STEP 9: Save TF-IDF Vectorizer

print("\n9. Saving TF-IDF model...")

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)

print("TF-IDF Vectorizer saved successfully!")


# STEP 10: Save Similarity Results

print("\n10. Saving similarity results...")

results = test_df[
    [
        "question1",
        "question2",
        "is_duplicate"
    ]
].copy()

results["similarity_score"] = scores

results.to_csv(
    "dataset/similarity_results.csv",
    index=False
)

print("Similarity results saved successfully!")


# STEP 11: Final Output

print("\n===================================")
print(" FEATURE ENGINEERING COMPLETED!")
print("===================================")

print("Total Dataset Records:", len(df))
print("Training Records:", len(train_df))
print("Testing Records:", len(test_df))

print("\nGenerated Files:")
print("1. model/tfidf_vectorizer.pkl")
print("2. dataset/similarity_results.csv")

print("\nYour feature engineering phase is complete!")