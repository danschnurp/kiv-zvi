# %%

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

import os

# A command line argument.
input_dir = "/Users/danschnurpfeil/PycharmProjects/kiv-zvi/SP/data_katastr/"

img_names = os.listdir(input_dir)
for i in img_names:
    print(i)
print(len(img_names), "pictures loaded")
input_name = input_dir + img_names[9]

print(input_name)

# %%
"""
##### LOADING BW
"""


# %%
def load_BW():
    return cv2.imread(input_name, cv2.IMREAD_GRAYSCALE)


# %%
"""
#### LOADING COLORED
"""


# %%
# It reads the image from the file.
def load_colored():
    # img = img[:,:,0]
    return cv2.cvtColor((cv2.imread(input_name, cv2.COLOR_BGR2RGB)), cv2.COLOR_BGR2RGB)


# %%
"""
- convolving image by horizontal line detection mask
"""


# %%
def convolve_horizontal_lightly(img):
    kernel_horizontal_lines = np.array([[-1, -1, -1],
                                        [2, 2, 2],
                                        [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel_horizontal_lines)


# %%
"""
- convolving image by vertical line detection mask
"""


# %%
def convolve_vertical_lightly(img):
    kernel_horizontal_lines = np.array([[-1, 2, -1],
                                        [-1, 2, -1],
                                        [-1, 2, -1]])

    return cv2.filter2D(img, -1, kernel_horizontal_lines)


# %%
"""
- A Canny edge detector 
"""

# %%
# img = cv2.Canny(img, 1, 500)
# ...
# kernel = np.ones((3,3), np.uint8)
# kernel[:,:] = 5
# print(kernel)
# img = cv2.dilate(img, kernel)
# img = cv2.erode(img, kernel)
# img = cv2.bitwise_not(img)

# %%
img = load_BW()

assert img is not None, "file could not be read, check with os.path.exists()"

# plt.imshow(img, cmap="gray")
# plt.show()

ret, thresh1 = cv2.threshold(img, 100, 155, cv2.THRESH_OTSU)

# plt.imshow(thresh1, cmap="gray")
img = thresh1


# %%
# sauvola = (cv2.imread(input_name, cv2.IMREAD_GRAYSCALE))
# assert img is not None, "file could not be read, check with os.path.exists()"
# sauvola = threshold_sauvola(img, window_size=5, k=0.1)

# plt.imshow(sauvola, cmap="gray")
# img = sauvola

# %%
def get_sizes_procentual(procentual: float):
    # Taking the right xx% of the image.
    right_border = int(img.shape[1] * procentual)
    # Taking the left xx% of the image.
    left_border = int(img.shape[1] * (1 - procentual))
    # Taking the bottom xx% of the image.
    bottom_border = int(img.shape[0] * procentual)
    # Taking the top xx% of the image.
    top_border = int(img.shape[0] * (1 - procentual))
    return right_border, left_border, bottom_border, top_border


# %%
"""
### cropping image by 1% from each side
"""

# %%
right_border, left_border, bottom_border, top_border = get_sizes_procentual(0.01)
print(bottom_border, top_border, right_border, left_border)
print(img.shape)
img = img[bottom_border:top_border, right_border:left_border]

# plt.imshow(img, cmap="gray")


# %%
# todo danger with spaghetti pictures works badly
# Taking the right 90% of the image.
right_border = int(img.shape[1] * 0.9)
# Taking the left 10% of the image.
left_border = int(img.shape[1] * 0.1)
# Taking the bottom 90% of the image.
bottom_border = int(img.shape[0] * 0.9)
# Taking the top 10% of the image.
top_border = int(img.shape[0] * 0.1)

# Taking the right, left, bottom and top border of the image.
right_border = img[:, right_border:]
left_border = img[:, :left_border]
bottom_border = img[bottom_border:, :]
top_border = img[:top_border, :]
print(top_border.shape)


# top_border = top_border.astype(int)
# top_border = img_as_ubyte(top_border)
# print(top_border[:5])

# %%
def euclidean_2D(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# %%


def get_vertical(x1, x2, y1, y2, theta=2, min_value=800):
    """
    computes if is line vertical or horisontal like by parameters theta as and min_value
    """

    if np.square(x1 - x2) < theta and np.square(y1 - y2) > min_value:
        return "vertical"
    elif np.square(y1 - y2) < theta and np.square(x1 - x2) > min_value:
        return "horizontal"
    else:
        return "other"


# def connect_lines(x1, x2, y1, y2, theta=2, min_value=200):
#     """
#     computes if is line vertical or horisontal like by parameters theta as and min_value
#     """
#
#     if np.abs(x1 - x2) < theta and np.abs(y1 - y2) > min_value:
#         return "vertical"
#     elif np.abs(y1 - y2) < theta and np.abs(x1 - x2) > min_value:
#         return "horizontal"
#     else:
#         return "other"


# %%

def LDS(border):
    # Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines in the image
    lines = lsd.detect(border)[0]  # Position 0 of the returned tuple are the detected lines
    lines = np.squeeze(lines)
    return lines


def hough(border):
    #  Detecting lines in the image with probabilistic_hough_line transform
    lines = probabilistic_hough_line(border, threshold=10, line_length=5,
                                     line_gap=3)
    for line in lines:
        p0, p1 = line
        # plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color="black")

    # plt.show()
    lines = [[i[0], i[1], j[0], j[1]] for i, j in lines]
    lines = np.array(lines)

    return lines


def plot_border_line(longest_lines, border):
    # Draw detected lines in the image
    for line_to_draw in longest_lines:
        border = cv2.line(border, (line_to_draw[0], line_to_draw[1]), (line_to_draw[2], line_to_draw[3]), (0))
    print(border.shape)
    # It converts the image from BGR to grayscale.
    # drawn_img = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    # plt.imshow(border, cmap="gray")
    return border


# %%


def make_line_detection(border, detection_method, img_shape, horizontal_only=True, vertical_only=False):
    if horizontal_only:
        for _ in range(9):
            border = convolve_horizontal_lightly(border)
    elif vertical_only:
        for _ in range(9):
            border = convolve_vertical_lightly(border)

    lines = detection_method(border)

    if vertical_only:
        min_value = img_shape[0] // 8
    else:
        min_value = img_shape[1] // 8

    directions = np.array([get_vertical(i[0], i[2], i[1], i[3], min_value=min_value) for i in lines])
    vertical = lines[directions == "vertical", :]
    print("vertical", vertical.shape)
    horizontal = lines[directions == "horizontal", :]
    print("horizontal", horizontal.shape)

    if horizontal_only:
        lines = horizontal

    elif vertical_only:
        lines = vertical

    # lines = connect_lines(*lines.T) todo

    # print(lines.shape)
    distances = np.array([euclidean_2D(i[0], i[2], i[1], i[3], ) for i in lines])
    print("distances.shape", distances.shape)

    print(distances.shape)
    # plt.plot(np.arange(0, len(distances)), distances)
    # plt.show()
    longest_lines = lines[distances > np.percentile(distances, 75)]
    # longest_lines = lines
    print("longest_lines.shape", longest_lines.shape)

    longest_lines = longest_lines.astype(int)
    #     print(longest_lines)

    border[:, :] = 255

    return plot_border_line(longest_lines=longest_lines, border=border)


# %%
right_border_line = make_line_detection(right_border, hough, horizontal_only=False, vertical_only=True, img_shape=img.shape)

# %%
left_border_line = make_line_detection(left_border, hough, horizontal_only=False, vertical_only=True, img_shape=img.shape)

# %%
top_border_line = make_line_detection(top_border, hough, horizontal_only=True, vertical_only=False, img_shape=img.shape)

# %%
bottom_border_line = make_line_detection(bottom_border, hough, horizontal_only=True, img_shape=img.shape)


