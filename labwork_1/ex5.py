from cv2 import (
    imread, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
    cvtColor, COLOR_BGR2GRAY
)
from numpy import dstack
import matplotlib.pyplot as plt

def threshold(image, th):
    return [[0 if cell < th else 255 for cell in row] for row in image]

img = imread('../resource/target.png')

img_gray = cvtColor(img, COLOR_BGR2GRAY)

print(img_gray)

plt.subplot(131)
plt.imshow(img_gray, cmap='gray')

plt.subplot(132)
plt.imshow(threshold(img_gray, 100), cmap='gray')

plt.subplot(133)
plt.imshow(adaptiveThreshold(img_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 12), cmap='gray')

plt.show()
