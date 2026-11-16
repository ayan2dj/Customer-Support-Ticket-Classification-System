
import pandas as pd

# Load dataset
df = pd.read_csv("data/tickets.csv")

# Display first five records
print("FIRST FIVE RECORDS:")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# Category distribution
print("\nCATEGORY DISTRIBUTION:")
print(df["category"].value_counts())

# Total records
print("\nTOTAL TICKETS:", len(df))
