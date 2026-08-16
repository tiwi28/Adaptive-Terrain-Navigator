import csv

files = ["hardwood.csv", "carpet.csv", "incline.csv"]

for f in files:
    with open(f) as file:
        rows = sum(1 for row in file) - 1  # minus header
    print(f"{f}: {rows} rows")