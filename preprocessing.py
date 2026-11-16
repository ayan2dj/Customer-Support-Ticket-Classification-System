
import pandas as pd
import re
import string

# ---------------------------------
# 1. Load Dataset
# ---------------------------------

df = pd.read_csv("data/tickets.csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Dataset Shape:", df.shape)


# ---------------------------------
# 2. Check Missing Values
# ---------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove records without essential information
df = df.dropna(
    subset=["ticket_id", "ticket_description", "category"]
)

# Fill optional columns
df["priority"] = df["priority"].fillna("Unknown")
df["status"] = df["status"].fillna("Unknown")


# ---------------------------------
# 3. Check Duplicate Records
# ---------------------------------

print("\nDuplicate Records:", df.duplicated().sum())

# Remove completely duplicated rows
df = df.drop_duplicates()

# Remove duplicate ticket IDs, keeping first record
df = df.drop_duplicates(
    subset=["ticket_id"],
    keep="first"
)


# ---------------------------------
# 4. Text Cleaning Function
# ---------------------------------

def clean_text(text):

    # Convert text into string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------
# 5. Apply Text Cleaning
# ---------------------------------

df["cleaned_description"] = (
    df["ticket_description"].apply(clean_text)
)

# Remove empty descriptions after cleaning
df = df[df["cleaned_description"] != ""]


# ---------------------------------
# 6. Display Cleaned Dataset
# ---------------------------------

print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Dataset Shape:", df.shape)


# ---------------------------------
# 7. Final Data Information
# ---------------------------------

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCategory Distribution:")
print(df["category"].value_counts())


# ---------------------------------
# 8. Save Cleaned Dataset
# ---------------------------------

df.to_csv(
    "data/cleaned_tickets.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
