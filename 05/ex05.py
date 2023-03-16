# %%
import cv2
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_image_cv2(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read..."
    return img


def show_image_now(image: np.ndarray, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.show()


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


img = load_image_cv2(path_to_images() + "son1.jpg")

show_image_now(img, cmap='gray')

# %%


ret, thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

show_image_now(thresh1)

# %%

