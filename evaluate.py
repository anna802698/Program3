import argparse
import joblib
import pandas as pd
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------
# Arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--features', required=False)
parser.add_argument('--output', required=True)

args = parser.parse_args()

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(args.dataset, sep=';')

# Encode target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# One-hot encoding
df = pd.get_dummies(df)

X = df.drop('y', axis=1)
y = df['y']

# -------------------------------
# Apply selected features
# -------------------------------
if args.features and os.path.exists(args.features):
    selected_features = pd.read_csv(args.features).iloc[:, 0].tolist()
    selected_features = [f for f in selected_features if f in X.columns]
    X = X[selected_features]

# -------------------------------
# Load model
# -------------------------------
model = joblib.load(args.model)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X)

# For ROC-AUC (needs probabilities)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)[:, 1]
else:
    # fallback for models like SVM (without probability=True)
    y_prob = y_pred

# -------------------------------
# Metrics
# -------------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

# Handle ROC-AUC safely
try:
    roc_auc = roc_auc_score(y, y_prob)
except:
    roc_auc = None

cm = confusion_matrix(y, y_pred).tolist()
report = classification_report(y, y_pred, output_dict=True)

# -------------------------------
# Save results
# -------------------------------
os.makedirs(os.path.dirname(args.output), exist_ok=True)

results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "confusion_matrix": cm,
    "classification_report": report
}

with open(args.output, "w") as f:
    json.dump(results, f, indent=4)

# -------------------------------
# Print summary (for logs)
# -------------------------------
print("===== Evaluation Results =====")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc}")
print("Confusion Matrix:", cm)