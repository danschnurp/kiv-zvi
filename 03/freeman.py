#  date: 04. 03. 2023
#  author: Daniel Schnurpfeil
#

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import path_to_images, show_image_now


def shift1(arr):
    """
    It shifts the elements of the array to the right by one.
    
    :param arr: the array to be shifted
    """
    return np.roll(arr, -1)


def print_area(cont):
    """
    This function takes a list of contours and prints the area of each contour.
    
    :param cont: the contour you want to find the area of
    """
    print("Contour detection (inner border) area")
    print(0.5 * np.sum((shift1(cont[:, 0]) - cont[:, 0]) * (cont[:, 1] + shift1(cont[:, 1])).T))


def get_4_neighbourhood(first_x: int, second_x: int, first_y: int, second_y: int) -> int:
    """
    > This function returns the number of neighbours of a cell in a 2D grid
    
    :param first_x: the x coordinate of the first point
    :type first_x: int
    :param second_x: the x coordinate of the second point
    :type second_x: int
    :param first_y: the y coordinate of the first point
    :type first_y: int
    :param second_y: the y coordinate of the second point
    :type second_y: int
    """
    if first_x > second_x:
        return 0

    if first_y > second_y:
        return 6

    if first_y < second_y:
        return 2

    if first_x < second_x:
        return 4


def compute_freeman_chaincode_4_neighbourhood(contours: np.ndarray):
    """
    It computes the Freeman chain code for each contour in the input array
    
    :param contours: a list of contours, each contour is a list of points, each point is a list of x,y
    coordinates
    :type contours: np.ndarray
    """
    shifted = np.roll(contours, 1, axis=0)
    return np.array([get_4_neighbourhood(x1, x2, y1, y2)
                     for x1, x2, y1, y2 in zip(contours[:, 0], shifted[:, 0], contours[:, 1], shifted[:, 1])])


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


if __name__ == "__main__":
    img = cv2.bitwise_not(cv2.imread(path_to_images() + "nula.png", cv2.IMREAD_GRAYSCALE))

    contour = getContour(img)
    contour = contour
    print(contour.shape)

    freeman_code = compute_freeman_chaincode_4_neighbourhood(np.squeeze(contour))
    plt.plot(np.arange(len(freeman_code)), freeman_code)
    plt.show()
