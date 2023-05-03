import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("albert.jpg", cv2.IMREAD_GRAYSCALE)

'''
compute FFT, show magnitude spectrum
'''
f = np.fft.fft2(img)
print(f.shape)
print("min", f.min())
print("max", f.max())

siz = 30

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('img_back'), plt.xticks([]), plt.yticks([])

plt.show()
