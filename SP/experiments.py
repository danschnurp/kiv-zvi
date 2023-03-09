import cv2
import numpy as np
from matplotlib import pyplot as plt

input_name = "UAZK-B2-a-04-C-1425-002.JPG"

img = (cv2.imread("./data_katastr/" + input_name, cv2.IMREAD_GRAYSCALE))

assert img is not None, "file could not be read, check with os.path.exists()"

edges = cv2.Canny(img, 1, 500)
plt.imshow(edges, cmap='gray')


# cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
#                            param1=50, param2=30, minRadius=0, maxRadius=0)
# circles = np.uint16(np.around(circles))
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     # draw the center of the circle
#     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
# cv2.imshow('detected circles', cimg)

plt.show()
