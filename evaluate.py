import argparse
import joblib
import json
import pandas as pd
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--output')
args = parser.parse_args()

model = joblib.load(args.model)
df = pd.read_csv(args.dataset)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

pred = model.predict(X)
acc = accuracy_score(y, pred)

with open(args.output, "w") as f:
    json.dump({"accuracy": acc}, f)