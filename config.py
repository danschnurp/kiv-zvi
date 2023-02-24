#  date: 21. 2. 2023
#  author: Daniel Schnurpfeil
#
import os


def path_to_images(output_filename="images", output_dir="../") -> str:
    if output_filename not in os.listdir(output_dir):
        if output_dir[-1] != "/":
            output_dir += "/"
        os.mkdir(output_dir + output_filename)
    output_dir += output_filename
    return output_dir + "/"
