import cv2 as cv

from numpy import log10, copysign


def bounus1():
    img = cv.imread("../images/dvojka.png", cv.IMREAD_GRAYSCALE)
    # Calculate Moments
    moments = cv.moments(img)
    # Calculate Hu Moments
    huMoments = cv.HuMoments(moments)

    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    print(huMoments)


bounus1()
