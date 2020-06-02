from cv2 import (
    imread, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
    cvtColor, COLOR_BGR2GRAY,
    calcHist, equalizeHist
)
from numpy import dstack
import matplotlib.pyplot as plt

def threshold(image, th):
    return [[0 if cell < th else 255 for cell in row] for row in image]

img = imread('../resource/target.png')

img_gray = cvtColor(img, COLOR_BGR2GRAY)

hist_img = calcHist([img_gray], [0], None, [256], [0, 255])

img_eq = img_gray

plt.subplot(131)
plt.imshow(img_gray, cmap='gray')

plt.subplot(132)
plt.bar(range(256), hist_img[:, 0])

plt.subplot(133)
plt.imshow(img_eq, cmap='gray')

plt.show()
