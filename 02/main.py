import cv2
import matplotlib.pyplot as plt
import numpy as np
import our_contours as op


def area(im):
    # plt.imshow(img)
    # plt.show()
    contour = op.getContour(im)
    # print(contour.shape)
    # print(contour[:5], contour[-5:])
    s = 0
    for i in range(1, len(contour)):
        x2 = contour[i][0]
        x1 = contour[i - 1][0]
        y2 = contour[i][1]
        y1 = contour[i - 1][1]
        s += (x2 - x1) * (y2 + y1)
    x2 = contour[0][0]
    x1 = contour[-1][0]
    y2 = contour[-1][1]
    y1 = contour[0][1]
    s += (x2 - x1) * (y2 + y1)

    print(s / 2)


def main(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    a = cv2.contourArea(contours[0])
    print(a)


def shapes_1(im):
    x, y, w, h = cv2.boundingRect(im)

    print(x, y, w, h)

    circle = cv2.minEnclosingCircle(op.getContour(im))
    print(circle)

    cv2.circle(im, int(circle[0][0]), int(circle[0][1]), int(circle[1]), 127)
    plt.imshow(im)
    plt.show()


def pic_operations():
    img1 = cv2.imread("images/son1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("images/son2.jpg", cv2.IMREAD_GRAYSCALE)

    # result = img1.astype(np.int32) - img2.astype(np.int32)
    # result += 129
    # result = result.astype(np.uint8)

    result = img1.astype(np.int32) / img2.astype(np.int32)
    # result += 129
    # result = result.astype(np.uint8)

    plt.imshow(result, cmap='gray')
    plt.show()


if __name__ == '__main__':
    img = cv2.bitwise_not(cv2.imread("dvojka.png", cv2.IMREAD_GRAYSCALE))
    # area(img)
    # main(img)
    # shapes_1(img)
    # pic_operations()
