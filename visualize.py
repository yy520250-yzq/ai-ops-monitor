#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 读取数据
df = pd.read_csv("metrics_with_anomaly.csv")
# 加载模型
model = joblib.load("model.pkl")
# 预测异常
df['anomaly'] = model.predict(df[['cpu', 'mem']])

normal = df[df['anomaly'] == 1]
anomaly = df[df['anomaly'] == -1]

# 图1：CPU 趋势 + 异常点
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['cpu'], label='CPU %', color='blue')
plt.scatter(anomaly.index, anomaly['cpu'], color='red', label='Anomaly', zorder=5)
plt.title('CPU Usage with Anomaly Detection')
plt.xlabel('Sample Index')
plt.ylabel('CPU %')
plt.legend()
plt.savefig('cpu_anomaly.png')
plt.close()

# 图2：内存趋势 + 异常点
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['mem'], label='Memory %', color='green')
plt.scatter(anomaly.index, anomaly['mem'], color='red', label='Anomaly', zorder=5)
plt.title('Memory Usage with Anomaly Detection')
plt.xlabel('Sample Index')
plt.ylabel('Memory %')
plt.legend()
plt.savefig('mem_anomaly.png')
plt.close()

# 图3：特征空间聚类（CPU vs MEM）
plt.figure(figsize=(8, 6))
plt.scatter(normal['cpu'], normal['mem'], label='Normal', alpha=0.6)
plt.scatter(anomaly['cpu'], anomaly['mem'], label='Anomaly', color='red', marker='x', s=100)
plt.title('Anomaly Clustering (CPU vs Memory)')
plt.xlabel('CPU %')
plt.ylabel('Memory %')
plt.legend()
plt.savefig('ml_cluster.png')
plt.close()

print("已生成: cpu_anomaly.png, mem_anomaly.png, ml_cluster.png")
