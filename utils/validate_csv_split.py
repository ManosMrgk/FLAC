import csv

def validate_no_overlap(file1_path, file2_path):
    # Read first CSV file
    with open(file1_path, 'r') as file1:
        reader1 = csv.reader(file1)
        header1 = next(reader1)  # Skip header
        rows1 = {tuple(row) for row in reader1}  # Use a set for faster lookup

    # Read second CSV file
    with open(file2_path, 'r') as file2:
        reader2 = csv.reader(file2)
        header2 = next(reader2)  # Skip header
        rows2 = {tuple(row) for row in reader2}  # Use a set for faster lookup

    # Find any overlapping rows
    overlapping_rows = rows1.intersection(rows2)

    if overlapping_rows:
        print(f"Validation Failed: {len(overlapping_rows)} overlapping row(s) found!")
        for row in overlapping_rows:
            print(row)
    else:
        print("Validation Passed: No overlapping rows found between the files.")

# File paths
output_80 = '../data/half_meta_data_filtered80.csv'
output_20 = '../data/half_meta_data_filtered20.csv'

# Validate the two CSV files
validate_no_overlap(output_80, output_20)
