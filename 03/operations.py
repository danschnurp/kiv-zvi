import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import path_to_images


def normalize(X):
    u = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return (X - u) / s, u, s


img = cv2.bitwise_not(cv2.imread(path_to_images() + "add1.jpg", cv2.IMREAD_GRAYSCALE))

img2 = cv2.bitwise_not(cv2.imread(path_to_images() + "add2.jpg", cv2.IMREAD_GRAYSCALE))

res = img.astype(np.int32) + img2.astype(np.int32)

# res[255 > res] = 255
res, _, _ = normalize(res)

plt.imshow(res.astype(np.uint8))
plt.show()
