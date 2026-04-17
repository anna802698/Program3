import argparse
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -------------------------------
# Parse arguments from CLI
# -------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, help="Path to dataset CSV")
parser.add_argument('--features', required=False, help="Selected features CSV")
parser.add_argument('--model', default='rf', help="Model type: rf / svm")
parser.add_argument('--epochs', default=10, help="(Optional for DL)")
parser.add_argument('--save_model', required=True, help="Path to save model")
parser.add_argument('--log', required=True, help="Log file path")

args = parser.parse_args()


# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(args.dataset)

# Assume last column is target
df = pd.read_csv(args.dataset, sep=';')  # IMPORTANT for bank.csv

# Convert target (y) to numeric
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Convert categorical features using one-hot encoding
df = pd.get_dummies(df)

# Split features & target
X = df.drop('y', axis=1)
y = df['y']

# -------------------------------
# Apply Feature Selection (if given)
# -------------------------------
if args.features and os.path.exists(args.features):
    selected_features = pd.read_csv(args.features).iloc[:, 0].tolist()
    X = X[selected_features]


# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# Choose Model
# -------------------------------
if args.model == 'rf':
    model = RandomForestClassifier(n_estimators=100)

elif args.model == 'svm':
    model = SVC()

else:
    raise ValueError("Unsupported model. Use 'rf' or 'svm'")


# -------------------------------
# Train Model
# -------------------------------
model.fit(X_train, y_train)


# -------------------------------
# Evaluate
# -------------------------------
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)


# -------------------------------
# Save Model
# -------------------------------
os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
joblib.dump(model, args.save_model)


# -------------------------------
# Save Logs
# -------------------------------
os.makedirs(os.path.dirname(args.log), exist_ok=True)

with open(args.log, "w") as f:
    f.write(f"Model: {args.model}\n")
    f.write(f"Accuracy: {accuracy}\n")


print("Training completed")
print(f"Accuracy: {accuracy}")