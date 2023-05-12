#  date: 4. 5. 2023
#  author: Daniel Schnurpfeil
#
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_sauvola
from skimage.transform import probabilistic_hough_line


# It reads the image from the file.
def load_BW(input_picture_path):
    return cv2.imread(input_picture_path, cv2.IMREAD_GRAYSCALE)


# It reads the image from the file.
def load_colored(input_picture_path):
    return cv2.cvtColor(cv2.imread(input_picture_path), cv2.COLOR_BGR2RGB)


def convolve_horizontal_lightly(img):
    """
    The function convolves an image horizontally with a light filter.
    """
    kernel_horizontal_lines = np.array([[-1, -1, -1],
                                        [2, 2, 2],
                                        [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel_horizontal_lines)


def convolve_vertical_lightly(img):
    """
    The function convolves an image vertically with a light filter.
    """
    kernel_horizontal_lines = np.array([[-1, 2, -1],
                                        [-1, 2, -1],
                                        [-1, 2, -1]])

    return cv2.filter2D(img, -1, kernel_horizontal_lines)


def get_sizes_percentual(procentual: float, img):
    """
    This function takes in a percentage value and an image and returns the size of the image as a percentage of the original
    size.
        """
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
    """
    The function "crop_one_percent" takes an image as input and crops 1% of the image from all four sides.

    :param img_to_crop: The input image that needs to be cropped
    """
    right_border_to_crop, left_border_to_crop, bottom_border_to_crop, top_border_to_crop = get_sizes_percentual(0.01,
                                                                                                            img_to_crop)
    # print(bottom_border_to_crop, top_border_to_crop, right_border_to_crop, left_border_to_crop)
    # print(img_to_crop.shape)
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
    """
    The function calculates the Euclidean distance between two points in a 2D plane.

    :param x1: The x-coordinate of the first point
    :param x2: The parameter x2 represents the x-coordinate of the second point in a 2D coordinate system
    :param y1: The y-coordinate of the first point in a 2D plane
    :param y2: The parameter `y2` represents the y-coordinate of the second point in a 2D coordinate system
    """
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

    lines = [[i[0], i[1], j[0], j[1]] for i, j in lines]
    lines = np.array(lines)

    return lines


def plot_border_line(longest_lines, border):
    """
    The function takes in the longest lines and a border and plots a border line to picture
    """
    # Draw detected lines in the image
    for line_to_draw in longest_lines:
        border = cv2.line(border, (line_to_draw[0], line_to_draw[1]), (line_to_draw[2], line_to_draw[3]), (255, 0, 0),
                          thickness=10)

    return border


def make_line_detection(border, detection_method, img_shape, horizontal_only=True,
                        min_longest_liner_means_percentile=95, min_longest_liner_extremes_percentile=30):
    """
    This function takes in parameters related to line detection and returns a line detection object.

    :param border: It is a parameter that specifies the size of the border around the image. This is useful for some line
    detection methods that require a certain amount of padding around the image
    :param detection_method: The method used for line detection, such as Hough Transform or LSD
    :param img_shape: img_shape is a tuple that represents the shape of the image. It contains two values: the height and
    width of the image in pixels. For example, if the image is 640 pixels wide and 480 pixels tall, the img_shape tuple
    would be (480, 640)
    :param horizontal_only: A boolean parameter that specifies whether to detect lines only in the horizontal direction. If
    set to True, the function will only detect horizontal lines. If set to False, the function will detect lines in both
    horizontal and vertical directions, defaults to True (optional)
    :param min_longest_liner_means_percentile: This parameter is used to set the minimum percentile value for the mean length
    of the longest lines detected by the line detection method. It determines the threshold for considering a line as a
    valid detection. For example, if this parameter is set to 95, it means that only lines with a mean length longer,
    defaults to 95 (optional)
    :param min_longest_liner_extremes_percentile: This parameter is used to set the minimum percentile value for the length
    of the longest line detected by the algorithm. Specifically, it refers to the percentile value of the distance between
    the two endpoints of the longest line. A higher value for this parameter will result in longer lines being detected by
    the algorithm, defaults to 30 (optional)
    """

    if horizontal_only:
        for _ in range(9):
            border = convolve_horizontal_lightly(border)
        # setting min line length as 1/8 of side length
        min_value = img_shape[1] // 8
    else:
        for _ in range(9):
            border = convolve_vertical_lightly(border)
        # setting min line length as 1/8 of side length
        min_value = img_shape[0] // 8
    lines = detection_method(border)

    directions = np.array([get_vertical(i[0], i[2], i[1], i[3], min_value=min_value) for i in lines])
    vertical = lines[directions == "vertical", :]
    horizontal = lines[directions == "horizontal", :]

    if horizontal_only:
        lines = horizontal
        print("horizontal")

    else:
        lines = vertical
        print("vertical")

    distances = np.array([euclidean_2D(i[0], i[2], i[1], i[3], ) for i in lines])

    longest_lines_for_means = lines[distances > np.percentile(distances, min_longest_liner_means_percentile)].T
    longest_lines = lines[distances > np.percentile(distances, min_longest_liner_extremes_percentile)].T

    # crating the longest line from longest_lines
    if horizontal_only:
        first = np.concatenate([longest_lines[0], longest_lines[2]])
        second = np.concatenate([longest_lines_for_means[1], longest_lines_for_means[1]])
        longest_line = np.array([[
            np.min(first),
            np.mean(second),
            np.max(first),
            np.mean(second)]], dtype=int)
    else:
        first = np.concatenate([longest_lines_for_means[0], longest_lines_for_means[2]])
        second = np.concatenate([longest_lines[1], longest_lines[1]])
        longest_line = np.array([[
            np.mean(first),
            np.min(second),
            np.mean(first),
            np.max(second),
        ]], dtype=int)

    print("longest_line", longest_line)

    return longest_line


def main(image_name, input_picture_path, out_dir_path, min_longest_liner_means_percentile=95,
         min_longest_liner_extremes_percentile=30):
    # loading a colored image from the specified file path and then extracting the third channel (blue channel)
    img = load_colored(input_picture_path).T[2].T

    assert img is not None, "file could not be read, check with os.path.exists()"

    # apply THRESH_OTSU todo bug
    ret, thresh1 = cv2.threshold(img, 150, 230, cv2.THRESH_BINARY)

    # plt.imshow(thresh1, cmap="gray")

    img = thresh1

    img = crop_one_percent(img)
    # cv2.imwrite(out_dir_path + image_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # return
    # plt.imshow(img, cmap="gray")

    right_border, left_border, bottom_border, top_border = get_borders_x_percent(img)

    # Taking the right, left, bottom and top border of the image.
    right_border = img[:, right_border:]
    left_border = img[:, :left_border]
    bottom_border = img[bottom_border:, :]
    top_border = img[:top_border, :]
    print(top_border.shape)

    lines = [
        make_line_detection(top_border, hough, horizontal_only=True, img_shape=img.shape,
                            min_longest_liner_means_percentile=min_longest_liner_means_percentile,
                            min_longest_liner_extremes_percentile=min_longest_liner_extremes_percentile),
        make_line_detection(bottom_border, hough, horizontal_only=True, img_shape=img.shape,
                            min_longest_liner_means_percentile=min_longest_liner_means_percentile,
                            min_longest_liner_extremes_percentile=min_longest_liner_extremes_percentile),
        make_line_detection(left_border, hough, horizontal_only=False,
                            img_shape=img.shape,
                            min_longest_liner_means_percentile=min_longest_liner_means_percentile,
                            min_longest_liner_extremes_percentile=min_longest_liner_extremes_percentile),
        make_line_detection(right_border, hough, horizontal_only=False,
                            img_shape=img.shape,
                            min_longest_liner_means_percentile=min_longest_liner_means_percentile,
                            min_longest_liner_extremes_percentile=min_longest_liner_extremes_percentile)

    ]
    # loading image where it will be written to
    img = load_colored(input_picture_path)
    img = crop_one_percent(img)

    right_border, left_border, bottom_border, top_border = get_borders_x_percent(img)

    parts = [img[:top_border, :],
             img[bottom_border:, :],
             img[:, :left_border],
             img[:, right_border:], ]

    for part, line in zip(parts, lines):
        part = plot_border_line(line, part)

    if out_dir_path[-1] != "/" or out_dir_path[-2] != '"\\':
        out_dir_path += "/"
    cv2.imwrite(out_dir_path + image_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
