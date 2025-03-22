import csv
import os

def filter_data(input_file, output_file):
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        return
    with open(input_file, "r", encoding="utf-8") as in_f:
        reader = csv.reader(in_f, delimiter="\t")
        with open(output_file, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["year", "season", "id", "name", "gender", "age", "sport", "gold_medals"])
            for row in reader:
                if any(not row[i].strip() for i in [2,3,4,5,6,7,8,9]):
                    continue
                writer.writerow([row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]])

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(script_dir, "..", "data", "athletes_info.csv")
output_file_path = os.path.join(script_dir, "..", "data", "filtered_data.csv")
filter_data(input_file_path, output_file_path)