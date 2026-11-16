
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1. Load Cleaned Dataset
# -----------------------------------

df = pd.read_csv("data/cleaned_tickets.csv")

sns.set_theme(style="whitegrid")

print("CUSTOMER SUPPORT TICKET ANALYSIS")

# -----------------------------------
# 2. Total Number of Tickets
# -----------------------------------

total_tickets = len(df)

print("\nTotal Number of Tickets:", total_tickets)


# -----------------------------------
# 3. Category Distribution
# -----------------------------------

category_counts = df["category"].value_counts()

print("\nTickets by Category:")
print(category_counts)


# -----------------------------------
# 4. Priority Distribution
# -----------------------------------

priority_counts = df["priority"].value_counts()

print("\nTickets by Priority:")
print(priority_counts)


# -----------------------------------
# 5. Status Distribution
# -----------------------------------

status_counts = df["status"].value_counts()

print("\nTickets by Status:")
print(status_counts)


# -----------------------------------
# 6. Chart 1: Category Distribution
# -----------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="category",
    hue="category",
    order=category_counts.index,
    palette="viridis",
    legend=False
)

plt.title("Customer Support Tickets by Category")
plt.xlabel("Ticket Category")
plt.ylabel("Number of Tickets")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "screenshots/category_distribution.png",
    dpi=300
)

plt.show()


# -----------------------------------
# 7. Chart 2: Priority Distribution
# -----------------------------------

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="priority",
    hue="priority",
    order=["Low", "Medium", "High"],
    palette="Set2",
    legend=False
)

plt.title("Customer Support Tickets by Priority")
plt.xlabel("Priority Level")
plt.ylabel("Number of Tickets")

plt.tight_layout()

plt.savefig(
    "screenshots/priority_distribution.png",
    dpi=300
)

plt.show()


# -----------------------------------
# 8. Chart 3: Status Distribution
# -----------------------------------

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="status",
    hue="status",
    order=["Open", "In Progress", "Closed"],
    palette="Set1",
    legend=False
)

plt.title("Customer Support Tickets by Status")
plt.xlabel("Ticket Status")
plt.ylabel("Number of Tickets")

plt.tight_layout()

plt.savefig(
    "screenshots/status_distribution.png",
    dpi=300
)

plt.show()


# -----------------------------------
# 9. Category vs Priority
# -----------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="category",
    hue="priority",
    hue_order=["Low", "Medium", "High"],
    palette="Set2"
)

plt.title("Category vs Priority Distribution")
plt.xlabel("Ticket Category")
plt.ylabel("Number of Tickets")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "screenshots/category_vs_priority.png",
    dpi=300
)

plt.show()


# -----------------------------------
# 10. Important Observations
# -----------------------------------

print("\nEDA OBSERVATIONS")

print("Most common category:",
      category_counts.idxmax())

print("Least common category:",
      category_counts.idxmin())

print("Most common priority:",
      priority_counts.idxmax())

print("Most common status:",
      status_counts.idxmax())

print("\nEDA COMPLETED SUCCESSFULLY!")
