import pandas as pd
import joblib

model = joblib.load("model.pkl")

df = pd.read_csv("metrics.csv")

predictions = model.predict(df)

for i, pred in enumerate(predictions):

    cpu = df.iloc[i]["cpu"]
    mem = df.iloc[i]["mem"]

    if pred == -1:
        print(f"[WARNING] anomaly detected | CPU={cpu}% MEM={mem}%")
    else:
        print(f"[INFO] normal | CPU={cpu}% MEM={mem}%")
