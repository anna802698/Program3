import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--features')
parser.add_argument('--model')
parser.add_argument('--epochs')
parser.add_argument('--save_model')
parser.add_argument('--log')
args = parser.parse_args()

df = pd.read_csv(args.dataset)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, args.save_model)