import serial
import sys
import csv

import time
start = time.time()

ser = serial.Serial('COM7', 38400, timeout=3.0)
label = sys.argv[1]

with open(f"{label}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "label"])

    while True:
        line = ser.readline().decode('utf-8').strip()
        parts = line.split()

        if len(parts) == 6:
            ax, ay, az, gx, gy, gz = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            writer.writerow([ax, ay, az, gx, gy, gz, label])
            print(f"Saved: {ax} {ay} {az} {gx} {gy} {gz} | {label}")

        if time.time() - start > 120:
            break

print("----DATA COLLECTION COMPLETED----")