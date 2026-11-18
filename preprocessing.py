import pandas as pd
import re
import os

# ==========================================
# STEP 1: TEXT CLEANING FUNCTION
# ==========================================

# Function to clean text
def clean_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert text into lowercase
    text = str(text).lower()

    # Expand common abbreviations
    replacements = {
        "nlp": "natural language processing",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "svm": "support vector machine"
    }

    # Replace abbreviations with full forms
    for short, full in replacements.items():
        text = re.sub(
            r'\b' + re.escape(short) + r'\b',
            full,
            text
        )

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ==========================================
# STEP 2: LOAD ORIGINAL DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("dataset/train.csv")

print("Dataset loaded successfully!")

print("Original Dataset Shape:", df.shape)


# ==========================================
# STEP 3: CHECK REQUIRED COLUMNS
# ==========================================

required_columns = ["question1", "question2"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# ==========================================
# STEP 4: CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")

print(df[["question1", "question2"]].isnull().sum())


# ==========================================
# STEP 5: CLEAN QUESTION 1
# ==========================================

print("\nCleaning Question 1...")

df["question1"] = df["question1"].apply(clean_text)


# ==========================================
# STEP 6: CLEAN QUESTION 2
# ==========================================

print("Cleaning Question 2...")

df["question2"] = df["question2"].apply(clean_text)


# ==========================================
# STEP 7: REMOVE EMPTY QUESTIONS
# ==========================================

df = df[
    (df["question1"].str.strip() != "") &
    (df["question2"].str.strip() != "")
]


# ==========================================
# STEP 8: REMOVE DUPLICATE ROWS
# ==========================================

print("\nRemoving duplicate rows...")

df = df.drop_duplicates()


# ==========================================
# STEP 9: SAVE CLEANED DATASET
# ==========================================

os.makedirs("dataset", exist_ok=True)

df.to_csv(
    "dataset/cleaned_questions.csv",
    index=False
)


# ==========================================
# STEP 10: DISPLAY FINAL RESULTS
# ==========================================

print("\n===================================")

print("PREPROCESSING COMPLETED SUCCESSFULLY!")

print("===================================")

print("Final Dataset Shape:", df.shape)

print("\nCleaned Dataset Saved Successfully!")

print("File Location: dataset/cleaned_questions.csv")