
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import create_models


# -----------------------------------
# 1. Load Cleaned Dataset
# -----------------------------------

df = pd.read_csv("data/cleaned_tickets.csv")

print("Original records:", len(df))


# -----------------------------------
# 2. Remove Duplicate Descriptions
# -----------------------------------

df = df.drop_duplicates(
    subset=["cleaned_description"]
).copy()

print("Unique ticket descriptions:", len(df))


# -----------------------------------
# 3. Define Input and Output
# -----------------------------------

X = df["cleaned_description"]
y = df["category"]

print("\nAvailable Categories:")
print(y.value_counts())


# -----------------------------------
# 4. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# -----------------------------------
# 5. Create Models
# -----------------------------------

models = create_models()

results = {}


# -----------------------------------
# 6. Train and Evaluate Models
# -----------------------------------

for name, model in models.items():

    print("\n" + "=" * 45)
    print("Training:", name)
    print("=" * 45)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Display results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": y_pred
    }


# -----------------------------------
# 7. Compare Both Models
# -----------------------------------

comparison = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": result["accuracy"],
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1 Score": result["f1_score"]
    }
    for name, result in results.items()
])

print("\nMODEL COMPARISON:")
print(comparison.to_string(index=False))


# -----------------------------------
# 8. Select Best Model
# -----------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = results[best_model_name]["model"]

print("\nBest Model:", best_model_name)


# -----------------------------------
# 9. Create Confusion Matrix
# -----------------------------------

y_best_pred = results[best_model_name]["predictions"]

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_best_pred,
    labels=labels
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.tight_layout()

Path("screenshots").mkdir(exist_ok=True)

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300
)

plt.show()


# -----------------------------------
# 10. Save Best Model
# -----------------------------------

Path("model").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "model/ticket_classifier.pkl"
)

# Save the selected model name
with open("model/best_model.txt", "w") as file:
    file.write(best_model_name)

# Save comparison results
comparison.to_csv(
    "model/model_comparison.csv",
    index=False
)

print("\nBest model saved successfully!")
print("Model:", best_model_name)
