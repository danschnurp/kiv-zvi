import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Takes black&white image as input, foreground objects must be white
returns list with x and y coordinates of the contour start
first coordinate is x, second y
'''


def getContourStart(img):
    start = [0, 0]
    while img[start[1], start[0]] == 0:
        # print (start)
        start[0] += 1
        if start[0] == img.shape[1]:
            start[0] = 0
            start[1] += 1
    return start


'''
returns neighbouring point to the given point in direction "direction"
'''


def getNext(point, direction):
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


'''
calculates inner border for given image
'''


def getContour(img):
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
    print("Contour detection (inner border)")
    img = cv2.bitwise_not(cv2.imread("../images/dvojka.png", cv2.IMREAD_GRAYSCALE))

    contour = getContour(img)
    print(contour.shape)
    print(contour[:5], contour[-5:])
