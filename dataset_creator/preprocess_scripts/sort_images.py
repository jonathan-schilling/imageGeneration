"""
Copies all usable images into a separate folder.
"""

import csv
import os
import shutil


def is_image_usable(row):

    # counts quality_bad, light_dark, dust_bad_ constructions_bad or snow
    image_bad_count = (
        int(row[3]) + int(row[5]) + int(row[7]) + int(row[9]) + int(row[10])
    )

    # counts light_medium, dust_medium or constructions_medium
    medium_count = int(row[4]) + int(row[6]) + int(row[8])

    if image_bad_count > 0:
        # at least one bad label exist
        return False

    if medium_count > 2:
        # too many medium labels
        return False

    if medium_count == 2 and int(row[2]) == 1:
        # quality_medium + 2 medium labels
        return False

    return True


input_folder = "<path-to-input-images>"
output_folder = "<path-to-output-folder>"
label_file = "<path-to-file-with-labels-without-header>"

copied_file_count = 0

with open(label_file) as csvfile:
    input = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)

    # format rows: 'img', 'quality_good', 'quality_medium', 'quality_bad',
    #              'light_medium', 'light_dark', 'dust_medium', 'dust_bad',
    #              'constructions_medium', 'constructions_bad', 'snow'
    for row in input:
        if is_image_usable(row):
            shutil.copyfile(
                os.path.join(input_folder, row[0]), os.path.join(output_folder, row[0]),
            )

            copied_file_count += 1

            print(
                "Copyed File: ",
                copied_file_count,
                "Filename:",
                row[0],
                end="\r",
                flush=True,
            )
