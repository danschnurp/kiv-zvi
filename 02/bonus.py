import cv2
import numpy as np

from config import path_to_images


def create_representation(img):
    """
    It takes an image and returns the Hu moments of the image

    :param img: The image to be processed
    :return: HuMoments
    """
    return cv2.HuMoments(cv2.moments(img))


if __name__ == '__main__':

    representations = np.zeros((15, 7), dtype=np.float32)

    for i in range(1, 16):
        # Creating a representation of the image.
        representations[i - 1] = np.squeeze(create_representation(
            cv2.bitwise_not(cv2.imread(
                # Reading the image from the path and converting it to grayscale.
                path_to_images() + str(i) + ".png", cv2.IMREAD_GRAYSCALE))))

    # print(representations)
    # exit(0)
    '''
    clustering
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(representations, 3, None, criteria, 10, flags)

    # A dictionary that maps the true labels of the images to the index of the image.
    true_labels = {"A": [0, 2, 3, 4, 9, 10], "X": [1, 5, 7, 11, 14], "K": [6, 8, 12, 13]}
    labels_dict = {}
    for index, i in enumerate(np.squeeze(labels)):
        if i not in labels_dict:
            labels_dict[i] = [index]
        else:
            labels_dict[i].append(index)
    print(true_labels)
    print(labels_dict)
