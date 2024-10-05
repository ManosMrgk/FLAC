import csv
import random

def split_csv(file_path, output1_path, output2_path, split_ratio=0.8):
    # Read the CSV file and store rows
    with open(file_path, 'r') as file:
        reader = list(csv.reader(file))
        header = reader[0]  # header
        rows = reader[1:]   # data rows

    # Shuffling
    random.shuffle(rows)

    # Split based on ratio
    split_index = int(len(rows) * split_ratio)
    rows_80 = rows[:split_index]
    rows_20 = rows[split_index:]

    with open(output1_path, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerow(header)
        writer.writerows(rows_80)

    with open(output2_path, 'w', newline='') as file2:
        writer = csv.writer(file2)
        writer.writerow(header)
        writer.writerows(rows_20)


input_csv = '../data/half_meta_data_filtered.csv'
output_80 = '../data/half_meta_data_filtered80.csv'
output_20 = '../data/half_meta_data_filtered20.csv'

split_csv(input_csv, output_80, output_20)

