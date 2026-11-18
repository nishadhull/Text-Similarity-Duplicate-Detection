import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Load cleaned dataset
print("Loading cleaned dataset...")

df = pd.read_csv("dataset/cleaned_questions.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# --------------------------------
# GRAPH 1: Duplicate vs Non-Duplicate
# --------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="is_duplicate")

plt.title("Duplicate vs Non-Duplicate Questions")
plt.xlabel("Question Category (0 = Non-Duplicate, 1 = Duplicate)")
plt.ylabel("Number of Question Pairs")

plt.savefig("screenshots/graph1_duplicates.png")
plt.show()


# --------------------------------
# GRAPH 2: Missing / Empty Values
# --------------------------------

empty_values = {
    "Question 1": (df["question1"].fillna("").str.strip() == "").sum(),
    "Question 2": (df["question2"].fillna("").str.strip() == "").sum()
}

plt.figure(figsize=(7, 5))

plt.bar(empty_values.keys(), empty_values.values())

plt.title("Empty Questions Analysis")
plt.xlabel("Question Columns")
plt.ylabel("Number of Empty Questions")

plt.savefig("screenshots/graph2_missing_values.png")
plt.show()


# --------------------------------
# GRAPH 3: Question Length Distribution
# --------------------------------

sample = df.sample(min(100000, len(df)), random_state=42)

sample["q1_length"] = sample["question1"].fillna("").str.split().str.len()

plt.figure(figsize=(8, 5))

sns.histplot(sample["q1_length"], bins=40, kde=True)

plt.title("Question Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.savefig("screenshots/graph3_question_length.png")
plt.show()


# --------------------------------
# GRAPH 4: Most Frequently Used Words
# --------------------------------

text = " ".join(
    df["question1"].fillna("").sample(
        min(50000, len(df)),
        random_state=42
    ).astype(str)
)

words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

word_counts = Counter(words)

common_words = word_counts.most_common(15)

words_list = [word[0] for word in common_words]
counts_list = [word[1] for word in common_words]

plt.figure(figsize=(10, 5))

plt.bar(words_list, counts_list)

plt.title("Top 15 Most Frequently Used Words")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("screenshots/graph4_common_words.png")
plt.show()


# --------------------------------
# GRAPH 5: Question Length Comparison
# --------------------------------

df["q1_length"] = df["question1"].fillna("").str.split().str.len()

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="is_duplicate", y="q1_length")

plt.title("Question Length by Duplicate Category")
plt.xlabel("0 = Non-Duplicate, 1 = Duplicate")
plt.ylabel("Number of Words")

plt.ylim(0, 40)

plt.savefig("screenshots/graph5_length_comparison.png")
plt.show()


print("\nEDA COMPLETED SUCCESSFULLY!")

print("\nAll five graphs saved inside screenshots folder.")