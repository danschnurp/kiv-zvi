import os


def path_to_images(output_filename="images", output_dir="../") -> str:
    """
    > This function returns the path to the images directory

    :param output_filename: The name of the folder that will be created to store the images, defaults to images (optional)
    :param output_dir: The directory to save the images to, defaults to ../ (optional)
    """
    if output_filename not in os.listdir(output_dir):
        if output_dir[-1] != "/":
            output_dir += "/"
        os.mkdir(output_dir + output_filename)
    output_dir += output_filename
    return output_dir + "/"
