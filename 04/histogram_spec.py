# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import load_image_cv2, path_to_images, show_image_now

img = load_image_cv2(path_to_images() + "nebula.png")

show_image_now(img, cmap='gray')

# %%

# create histogram using opencv
# check its shape

hist = cv2.calcHist([img], [0], None, [256], [0, 255])
print(hist.shape)
hist = np.reshape(hist, (hist.shape[0],))
hist /= img.shape[0] * img.shape[1]
print(hist.shape)
plt.plot(np.arange(0, len(hist)), hist, color='blue')
plt.xlim([0, 256])
plt.title("First Histogram")
plt.show()

# %%

# prepare cumulative histogram


hist_sum = np.zeros_like(hist)

for i in range(1, len(hist)):
    hist_sum[i] = hist_sum[i - 1] + hist[i]

plt.title("Cumulative Histogram")

plt.plot(np.arange(0, len(hist_sum)), hist_sum, color='blue')

plt.xlim([0, 256])
plt.show()


# %%
def gauss(x, sigma, mean):
    val = 1 / (sigma * np.sqrt(2 * np.pi))
    val *= np.exp(-0.5 * ((x - mean) / sigma * (x - mean) / sigma))
    return val


# %%
target_hist = np.zeros(hist.shape)
for i in range(0, 256):
    target_hist[i] = gauss(i / 255, 0.1, 0.3) / 255

print(np.sum(target_hist))
plt.plot(target_hist, color='blue')
plt.xlim([0, 256])
plt.show()

# %%

# cumulative target histogram

target_hist_sum = np.zeros_like(target_hist)
for i in range(len(target_hist)):
    target_hist_sum[i] = target_hist_sum[i - 1] + target_hist[i]

plt.plot(target_hist_sum, color='blue')
plt.xlim([0, 256])
plt.title("Cumulative Target Histogram")
plt.show()

# %%

# prepare lookup table for mapping

mapping = np.zeros(hist.shape)

target_val = 0

for i in range(len(mapping)):
    mapping[i] = np.argmax(target_hist_sum > hist_sum[i])
# editing negative burnings
mapping[np.argmax(mapping):] = 255


print(mapping[:10], mapping[-10:])

# %%

# remap the image


new_img = np.zeros_like(img)

# mapping by vectors (rows)
for index, i in enumerate(img):
    new_img[index] = mapping[i]

plt.imshow(new_img, cmap='gray')

# %%

# check resulting histogram

hist = cv2.calcHist([new_img], [0], None, [256], [0, 255])
print(hist.shape)
hist = np.reshape(hist, (hist.shape[0],))
hist /= img.shape[0] * img.shape[1]
print(hist.shape)
plt.plot(hist, color='blue')
plt.xlim([0, 256])
plt.show()

# %%
