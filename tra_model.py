import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("metrics.csv")

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(df)

joblib.dump(model, "model.pkl")

print("model trained")
