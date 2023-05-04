import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import probabilistic_hough_line

def load_BW(input_picture_path):
    return cv2.imread(input_picture_path, cv2.IMREAD_GRAYSCALE)



# It reads the image from the file.
def load_colored(input_picture_path):
    # img = img[:,:,0]
    return cv2.cvtColor((cv2.imread(input_picture_path, cv2.COLOR_BGR2RGB)), cv2.COLOR_BGR2RGB)



def convolve_horizontal_lightly(img):
    kernel_horizontal_lines = np.array([[-1, -1, -1],
                                        [2, 2, 2],
                                        [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel_horizontal_lines)




def convolve_vertical_lightly(img):
    kernel_horizontal_lines = np.array([[-1, 2, -1],
                                        [-1, 2, -1],
                                        [-1, 2, -1]])

    return cv2.filter2D(img, -1, kernel_horizontal_lines)


def get_sizes_procentual(procentual: float, img):
    # Taking the right xx% of the image.
    right_border = int(img.shape[1] * procentual)
    # Taking the left xx% of the image.
    left_border = int(img.shape[1] * (1 - procentual))
    # Taking the bottom xx% of the image.
    bottom_border = int(img.shape[0] * procentual)
    # Taking the top xx% of the image.
    top_border = int(img.shape[0] * (1 - procentual))
    return right_border, left_border, bottom_border, top_border


def crop_one_percent(img_to_crop):
    right_border_to_crop, left_border_to_crop, bottom_border_to_crop, top_border_to_crop = get_sizes_procentual(0.01,
                                                                                                                img_to_crop)
    print(bottom_border_to_crop, top_border_to_crop, right_border_to_crop, left_border_to_crop)
    print(img_to_crop.shape)
    return img_to_crop[bottom_border_to_crop:top_border_to_crop, right_border_to_crop:left_border_to_crop]


def get_borders_x_percent(img_input, x=10):
    # todo danger with spaghetti pictures works badly
    x /= 100
    # Taking the right x% of the image.
    # Taking the left x% of the image.
    # Taking the bottom x% of the image.
    # Taking the top x% of the image.
    return int(img_input.shape[1] * (1 - x)), \
           int(img_input.shape[1] * x), \
           int(img_input.shape[0] * (1 - x)), \
           int(img_input.shape[0] * x)


def euclidean_2D(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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
#     pass


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
    # for line in lines:
    #     p0, p1 = line
    # plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color="black")

    # plt.show()
    lines = [[i[0], i[1], j[0], j[1]] for i, j in lines]
    lines = np.array(lines)

    return lines


def plot_border_line(longest_lines, border):
    # Draw detected lines in the image
    for line_to_draw in longest_lines:
        border = cv2.line(border, (line_to_draw[0], line_to_draw[1]), (line_to_draw[2], line_to_draw[3]), (255, 0, 0),
                          thickness=100)
    print(border.shape)
    # kernel = np.zeros((3, 3))
    # kernel[1] = -1
    # kernel.T[1] = -1
    # kernel[1, 1] = 5
    # # plt.imshow(top_border, cmap="gray")
    # border = cv2.filter2D(border, -1, kernel)
    # plt.imshow(border, cmap="gray")
    # plt.show()

    return border


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

    return longest_lines


def main(image_name, input_picture_path):
    img = load_BW(input_picture_path)

    assert img is not None, "file could not be read, check with os.path.exists()"

    # plt.imshow(img, cmap="gray")
    # plt.show()

    ret, thresh1 = cv2.threshold(img, 100, 155, cv2.THRESH_OTSU)

    # plt.imshow(thresh1, cmap="gray")
    img = thresh1

    # sauvola = (cv2.imread(input_name, cv2.IMREAD_GRAYSCALE))
    # assert img is not None, "file could not be read, check with os.path.exists()"
    # sauvola = threshold_sauvola(img, window_size=5, k=0.1)

    # plt.imshow(sauvola, cmap="gray")
    # img = sauvola

    img = crop_one_percent(img)

    # plt.imshow(img, cmap="gray")

    right_border, left_border, bottom_border, top_border = get_borders_x_percent(img)

    # Taking the right, left, bottom and top border of the image.
    right_border = img[:, right_border:]
    left_border = img[:, :left_border]
    bottom_border = img[bottom_border:, :]
    top_border = img[:top_border, :]
    print(top_border.shape)

    lines = [
        make_line_detection(top_border, hough, horizontal_only=True, vertical_only=False, img_shape=img.shape),
        make_line_detection(bottom_border, hough, horizontal_only=True, img_shape=img.shape),
        make_line_detection(left_border, hough, horizontal_only=False, vertical_only=True,
                            img_shape=img.shape),
        make_line_detection(right_border, hough, horizontal_only=False, vertical_only=True,
                            img_shape=img.shape)

    ]

    img = load_colored(input_picture_path)
    img = crop_one_percent(img)

    right_border, left_border, bottom_border, top_border = get_borders_x_percent(img)

    parts = [img[:top_border, :],
             img[bottom_border:, :],
             img[:, :left_border],
             img[:, right_border:], ]

    for part, line in zip(parts, lines):
        part = plot_border_line(line, part)
    plt.imshow(img, cmap="gray")
    plt.savefig("./annotated_katastr/" + image_name)


import os

input_dir = "/Users/danschnurpfeil/PycharmProjects/kiv-zvi/SP/data_katastr/"

img_names = os.listdir(input_dir)
for image_name in img_names:
    input_picture_path = input_dir + image_name
    main(image_name=image_name, input_picture_path=input_picture_path)
