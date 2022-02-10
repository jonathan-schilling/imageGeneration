"""
Check validity of the labeled images.
"""

import csv


def is_row_valid(row):
    """
    Checks if an image has maximal one label of each category QUALITY, LIGHT, DUST, CONSTRUCTIONS.
    """

    quality = int(row[1]) + int(row[2]) + int(row[3])
    light = int(row[4]) + int(row[5])
    dust = int(row[6]) + int(row[7])
    constructions = int(row[8]) + int(row[9])

    if quality != 1 or light > 1 or dust > 1 or constructions > 1:
        return False
    else:
        return True

label_file = "<path-to-file-with-labels-without-header>"

with open(label_file) as csvfile:
    input = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)

    num_invalid_rows = 0
    invalid_rows = []

    # format rows: 'img', 'quality_good', 'quality_medium', 'quality_bad',
    #              'light_medium', 'light_dark', 'dust_medium', 'dust_bad',
    #              'constructions_medium', 'constructions_bad', 'snow'
    for row in input:
        if not is_row_valid(row):
            num_invalid_rows = num_invalid_rows + 1
            invalid_rows.append(row[0])

    if num_invalid_rows == 0:
        print("CSV file is valid.")
    else:
        if num_invalid_rows == 1:
            print("CSV file isn't valid, there is", num_invalid_rows, "invalid row.")
            print("The invalid row correspond to image:", invalid_rows[0])
        else:
            print("CSV file isn't valid, there are", num_invalid_rows, "invalid rows.")
            print("The invalid rows correspond to images:", invalid_rows)
