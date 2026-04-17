import argparse
import pandas as pd

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('--dataset', required=True)
parser.add_argument('--method', default='clo', help="Feature selection method")
parser.add_argument('--output', required=True, help="Output file")

args = parser.parse_args()


# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(args.dataset, sep=';')

# Convert target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# One-hot encoding
df = pd.get_dummies(df)

X = df.drop('y', axis=1)


# -------------------------------
# Dummy Feature Selection (Replace with CLO/AOOA)
# -------------------------------
selected_features = X.columns[:5]  # pick first 5 for now


# -------------------------------
# Save selected features
# -------------------------------
pd.DataFrame(selected_features).to_csv(args.output, index=False)

print(f"Selected features saved to {args.output}")