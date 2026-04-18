import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("data/dataset-tickets-multi-lang-4-20k.csv")

# -----------------------------
# Step 2: Preprocess
# -----------------------------

# Combine subject and body into one text column (clean spacing)
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

# Keep only required columns
df = df[["text", "queue", "priority", "answer"]]

# Drop rows where important columns are missing
df = df.dropna(subset=["text", "queue", "priority"])

# Reset index
df = df.reset_index(drop=True)

print("Total samples after cleaning:", len(df))

# -----------------------------
# Step 3: Split data (80/10/10)
# -----------------------------

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: split temp into 10% val and 10% test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# -----------------------------
# Step 4: Save files
# -----------------------------

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# -----------------------------
# Optional: Quick sanity check
# -----------------------------
print("\nSample data:")
print(train_df.head())

print("\nIssue distribution:")
print(train_df["queue"].value_counts())

print("\nUrgency distribution:")
print(train_df["priority"].value_counts())