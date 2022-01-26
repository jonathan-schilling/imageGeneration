"""
Creates the input file for the open_images_downloader from a file of ids.
"""

import json

split = "train"

with open("kyd_ids.json") as f:
    data = json.load(f)
    ids = data["ids"]

with open("images_ids.txt", "w") as f:
    for id in ids:
        f.write(split + "/" + id[:-4] + "\n")
