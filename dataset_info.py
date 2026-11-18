import pandas as pd

# 1. Load the dataset
df = pd.read_csv("dataset/train.csv")

# 2. Display first five rows
print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

# 3. Display dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 4. Display dataset shape
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# 5. Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 6. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 7. Count duplicate and non-duplicate pairs
print("\n========== DUPLICATE COUNTS ==========")
print(df["is_duplicate"].value_counts())