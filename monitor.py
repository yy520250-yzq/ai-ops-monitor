#!/usr/bin/env python3
import psutil
import csv
import time
import sys
from datetime import datetime

# 采集间隔（秒）
INTERVAL = 5
# 最大采集次数（设为0表示无限，可以手动 Ctrl+C 停止）
MAX_RECORDS = 100

def collect():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    return cpu, mem

def main():
    with open('metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'cpu', 'mem'])
        count = 0
        try:
            while MAX_RECORDS == 0 or count < MAX_RECORDS:
                ts = datetime.now().isoformat()
                cpu, mem = collect()
                writer.writerow([ts, cpu, mem])
                f.flush()
                print(f"[{ts}] CPU={cpu}% MEM={mem}%")
                count += 1
                time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\n采集停止，数据已保存到 metrics.csv")

if __name__ == '__main__':
    main()
