import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import path_to_images, show_image_now


def shift1(arr):
    return np.roll(arr, -1)


def print_area(cont):
    print("Contour detection (inner border) area")
    print(0.5 * np.sum((shift1(cont[:, 0]) - cont[:, 0]) * (cont[:, 1] + shift1(cont[:, 1])).T))


def get_4_neighbourhood(first_x: int, second_x: int, first_y: int, second_y: int) -> int:
    if first_x < second_x:
        return 0

    if first_y < second_y:
        return 6

    if first_y > second_y:
        return 2

    if first_x > second_x:
        return 4


def compute_freeman_chaincode_4_neighbourhood(contours: np.ndarray):
    shifted = np.roll(contours, 1, axis=0)
    return np.array([get_4_neighbourhood(x1, x2, y1, y2)
                     for x1, x2, y1, y2 in zip(contours[:, 0], shifted[:, 0], contours[:, 1], shifted[:, 1])])


if __name__ == "__main__":
    img = cv2.bitwise_not(cv2.imread(path_to_images() + "nula.png", cv2.IMREAD_GRAYSCALE))

    show_image_now(img)
    contour, _ = cv2.findContours(img, 1, 2)
    contour = contour[1]
    print(contour.shape)

    freeman_code = compute_freeman_chaincode_4_neighbourhood(np.squeeze(contour))
    plt.plot(np.arange(len(freeman_code)), freeman_code)
    plt.show()

    x, y, w, h = cv2.boundingRect(contour)

    show_image_now(cv2.rectangle(img, (x, y), (x + w, y + h), 128, 2))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    show_image_now(cv2.drawContours(img, [np.squeeze(np.array(box, dtype=int))], 0, 100, 2))
