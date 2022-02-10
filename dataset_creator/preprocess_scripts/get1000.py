"""
Partitionates a big dataset into subsets of 1000 images.
"""

import os
import shutil

input_folder = "<path-to-folder-with-input-images>"
output_folder = "<path-to-output-folder>"

folder_count = 0

for i, file in enumerate(os.listdir(input_folder)):
    if i % 1000 == 0:
        folder_count += 1
        os.mkdir(os.path.join(output_folder, str(folder_count)))
    shutil.copyfile(
        os.path.join(input_folder, file),
        os.path.join(output_folder, str(folder_count), file),
    )

    print(
        "Copyed File: ",
        i,
        "Folder:",
        folder_count,
        "Filename:",
        file,
        end="\r",
        flush=True,
    )
