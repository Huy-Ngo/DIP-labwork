from cv2 import (
    imread, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
    cvtColor, COLOR_BGR2GRAY, COLOR_BGR2RGB, inRange
)
from numpy import dstack, array
import matplotlib.pyplot as plt

def threshold(image, th):
    return [[0 if cell < th else 255 for cell in row] for row in image]

img = imread('../resource/target.png')

img_gray = cvtColor(img, COLOR_BGR2GRAY)
img_rgb = cvtColor(img, COLOR_BGR2RGB)

img_rgb_threshold = inRange(img_rgb, array([10, 0, 5]), array([150, 175, 200]))

plt.subplot(221)
plt.title('Original image')
plt.imshow(img_rgb)

plt.subplot(222)
plt.title('Thresholded color image')
plt.imshow(img_rgb_threshold, cmap='gray')

plt.subplot(223)
plt.title('Simple thresholding (gray)')
plt.imshow(threshold(img_gray, 100), cmap='gray')

plt.subplot(224)
plt.title('Adaptive threshold (gray)')
plt.imshow(adaptiveThreshold(img_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 12), cmap='gray')

plt.show()
