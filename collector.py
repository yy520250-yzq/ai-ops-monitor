import random
import pandas as pd

data = []

for i in range(200):

    cpu = random.randint(20, 40)
    mem = random.randint(30, 50)

    # 模拟异常
    if i > 180:
        cpu = random.randint(80, 95)
        mem = random.randint(85, 95)

    data.append({
        "cpu": cpu,
        "mem": mem
    })

df = pd.DataFrame(data)

df.to_csv("metrics.csv", index=False)

print("metrics.csv generated")
