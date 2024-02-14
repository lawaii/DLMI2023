import csv
import os

csv_file = 'HAM10000_metadata.csv'
csv_reader = csv.reader(open(csv_file))
for line in csv_reader:
    if os.path.exists("./images/" + line[1] + ".jpg"):
        os.rename("./images/" + line[1] + ".jpg", "./images/" + line[1] + "_" + line[2] + ".jpg")

