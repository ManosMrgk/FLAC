import csv
import random

def split_csv(input_file, output_file_80, output_file_20, split_ratio=0.8):
    # Read the CSV file
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        reader = list(csv.reader(csv_file))
        header = reader[0]  # Extract header
        rows = reader[1:]  # Extract rows

    # Shuffle rows randomly
    random.shuffle(rows)

    # Split the rows into 80% and 20%
    split_index = int(len(rows) * split_ratio)
    rows_80 = rows[:split_index]
    rows_20 = rows[split_index:]

    # Write the 80% split to a new file
    with open(output_file_80, 'w', newline='', encoding='utf-8') as out_80:
        writer = csv.writer(out_80)
        writer.writerow(header)  # Write header
        writer.writerows(rows_80)  # Write rows

    # Write the 20% split to another file
    with open(output_file_20, 'w', newline='', encoding='utf-8') as out_20:
        writer = csv.writer(out_20)
        writer.writerow(header)  # Write header
        writer.writerows(rows_20)  # Write rows

    print(f"File successfully split: {len(rows_80)} rows in {output_file_80}, {len(rows_20)} rows in {output_file_20}.")

# Example usage
split_csv('data/meta_data_filtered.csv', 'data/meta_data_filtered80.csv', 'data/meta_data_filtered20.csv')
