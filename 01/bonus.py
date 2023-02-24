"""
scripty pro predmet ZVI
...
 @author: Daniel Schnurpfeil
"""
import cv2 as cv
from matplotlib import pyplot as plt

from config import path_to_images

dog = path_to_images() + 'dog.jpg'
example = path_to_images() + 'example.png'


def bonus2():
    img = cv.imread(example, cv.IMREAD_COLOR)
    print(img.shape)
    print(img.dtype)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    prah = 150
    for i in range(len(img[0])):
        for j in range(len(img[1])):
            if img[i, j, [0, 0, 0]][0] < prah and img[i, j, [0, 0, 0]][1] < prah and img[i, j, [0, 0, 0]][2] < prah:
                img[i, j, [0, 0, 0]] = [255, 255, 255]
                img[i, j, [1, 1, 1]] = [255, 255, 255]
                img[i, j, [2, 2, 2]] = [255, 255, 255]

    plt.imshow(img)
    plt.show()
    return 0


def bonus():
    img = cv.imread(dog, cv.IMREAD_COLOR)
    print(img.shape)
    print(img.dtype)

    img = img[:, :, [0, 0, 0]]
    print(img.shape)
    plt.imshow(img)
    plt.show()
    return 0


def prahovani():
    img = cv.imread(dog, cv.IMREAD_GRAYSCALE)
    print(img.shape)
    print(img.dtype)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img[img < 120] = 0
    img[img > 120] = 255
    plt.imshow(img, cmap='gray')
    plt.show()
    # cv.imwrite("dogo2.jpg", r)
    return 0


def main():
    img = cv.imread(dog, cv.IMREAD_COLOR)
    print(img.shape)
    print(img.dtype)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    r, g, b = cv.split(img)
    r = img[50:400, 50:100, :]
    plt.imshow(r)
    plt.show()
    # cv.imwrite("dogo2.jpg", r)
    return 0


if __name__ == '__main__':
    # main()
    bonus()
    bonus2()
