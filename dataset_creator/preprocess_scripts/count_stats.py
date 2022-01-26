"""
Creates a statistic about the label distribution in the dataset
"""

import csv
import json

label_file = "<path-to-file-with-labels-without-header>"
stats_file = "<path-to-file-with-stats>"

with open(label_file) as csvfile:
    input = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)

    quality_good, quality_medium, quality_bad = (0,) * 3
    good_unique, good_one_med, good_two_med = (0,) * 3
    medium_unique, medium_one_med = (0,) * 2

    # format rows: 'img', 'quality_good', 'quality_medium', 'quality_bad',
    #              'light_medium', 'light_dark', 'dust_medium', 'dust_bad',
    #              'constructions_medium', 'constructions_bad', 'snow'
    for row in input:
        quality_good = quality_good + int(row[1])
        quality_medium = quality_medium + int(row[2])
        quality_bad = quality_bad + int(row[3])

        # counts quality_bad, light_dark, dust_bad_ constructions_bad or snow
        image_bad_count = (
            int(row[3]) + int(row[5]) + int(row[7]) + int(row[9]) + int(row[10])
        )

        # counts light_medium, dust_medium or constructions_medium
        medium_count = int(row[4]) + int(row[6]) + int(row[8])

        if image_bad_count > 0:
            # at least one bad label exist
            continue

        if medium_count > 2:
            # too many medium labels
            continue

        if medium_count == 0:
            if int(row[1]) == 1:
                # unique quality_good
                good_unique = good_unique + 1
            else:
                # unique quality_medium
                medium_unique = medium_unique + 1
            continue

        if medium_count == 1:
            if int(row[1]) == 1:
                # quality_good + 1 medium label
                good_one_med = good_one_med + 1
            else:
                # quality_medium + 1 medium label
                medium_one_med = medium_one_med + 1
            continue

        if medium_count == 2 and int(row[1]) == 1:
            # quality_good + 2 medium labels
            good_two_med = good_two_med + 1

    sum_images = quality_good + quality_medium + quality_bad
    sum_usable = (
        good_unique + good_one_med + good_two_med + medium_unique + medium_one_med
    )

    stats = {
        "sum_images": sum_images,
        "quality_good": quality_good,
        "quality_medium": quality_medium,
        "quality_bad": quality_bad,
        "sum_usable": sum_usable,
        "good_unique": good_unique,
        "good_one_med": good_one_med,
        "good_two_med": good_two_med,
        "medium_unique": medium_unique,
        "medium_one_med": medium_one_med,
    }

    with open(stats_file, "w") as file:
        file.write(json.dumps(stats, indent=4, sort_keys=False))
