import time
from copy import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import path_to_images


def getContourStart(img):
    '''
    Takes black&white image as input, foreground objects must be white
    returns list with x and y coordinates of the contour start
    first coordinate is x, second y
    '''
    start = [0, 0]
    while img[start[1], start[0]] == 0:
        # print (start)
        start[0] += 1
        if start[0] == img.shape[1]:
            start[0] = 0
            start[1] += 1
    return start


def getNext(point, direction):
    '''
    returns neighbouring point to the given point in direction "direction"
    '''
    next = point.copy()
    if direction == 0:
        next[0] += 1
    if direction == 2:
        next[0] -= 1
    if direction == 1:
        next[1] -= 1
    if direction == 3:
        next[1] += 1
    return next


def getContour(img):
    '''
    calculates inner border for given image
    '''
    start = getContourStart(img)
    print("Contour start:", start)
    borderPoints = []
    borderPoints.append(start)
    aktPoint = start.copy()
    direction = 3
    while True:
        direction = (direction + 3) % 4
        next = getNext(aktPoint, direction)
        while img[next[1], next[0]] == 0:
            direction = (direction + 1) % 4
            next = getNext(aktPoint, direction)
        aktPoint = next

        if aktPoint == start:
            break

        borderPoints.append(aktPoint)

    return np.array(borderPoints)


def shift1(arr):
    return np.roll(arr, -1)


def area(cont):
    print("Contour detection (inner border) area")
    return 0.5 * np.sum((shift1(cont[:, 0]) - cont[:, 0]) * (cont[:, 1] + shift1(cont[:, 1])).T)


def show_image_now(image: np.ndarray, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.show()


def get_4_neighbourhood(first_x: int, second_x: int, first_y: int, second_y: int) -> int:
    if first_x < second_x:
        return 0

    if first_y < second_y:
        return 6

    if first_y > second_y:
        return 2

    if first_x > second_x:
        return 4


def compute_freeman_chaincode(contours: np.ndarray):
    shifted = np.roll(contours, -1, axis=0)

    result = np.zeros(shifted.shape[0])
    result = np.array([get_4_neighbourhood(x1, x2, y1, y2)
                       for x1, x2, y1, y2 in zip(contours[:, 0], shifted[:, 0], contours[:, 1], shifted[:, 1])])
    return


if __name__ == "__main__":
    img = cv2.bitwise_not(cv2.imread(path_to_images() + "nula.png", cv2.IMREAD_GRAYSCALE))

    show_image_now(img)
    contour, _ = cv2.findContours(img, 1, 2)
    contour = contour[1]
    print(contour.shape)

    freeman_code = compute_freeman_chaincode(np.squeeze(contour))

    x, y, w, h = cv2.boundingRect(contour)

    show_image_now(cv2.rectangle(img, (x, y), (x + w, y + h), 128, 2))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    show_image_now(cv2.drawContours(img, [np.squeeze(np.array(box, dtype=int))], 0, 100, 2))
    # print("area", cv2.contourArea(contour))
    # tmp = area(contour)
    # print(tmp)
