#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# 读取渐变异常数据
df = pd.read_csv("metrics_realistic_anomaly.csv")

# 特征
X = df[['cpu', 'mem']].values

# 训练模型（污染率 = 异常点数/总点数 ≈ 30/230 ≈ 0.13）
model = IsolationForest(contamination=0.13, random_state=42)
model.fit(X)

# 保存模型
joblib.dump(model, "model_realistic.pkl")
print("✅ 模型已保存为 model_realistic.pkl")

# 添加预测列并显示前几行
df['anomaly'] = model.predict(X)
print(df[['cpu', 'mem', 'anomaly']].head(10))
