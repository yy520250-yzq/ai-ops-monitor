#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# 读取包含异常的数据集
df = pd.read_csv("metrics_with_anomaly.csv")

# 只使用 CPU 和 MEM 作为特征
X = df[['cpu', 'mem']].values

# 训练模型（污染率设为 0.1，因为数据中有 10 个异常点共 110 条）
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# 保存模型
joblib.dump(model, "model.pkl")
print("✅ 模型已保存为 model.pkl")

# 预览结果
df['anomaly'] = model.predict(X)
print(df[['cpu', 'mem', 'anomaly']].head(10))

