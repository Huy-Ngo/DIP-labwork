from cv2 import (
    imread, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
    cvtColor, COLOR_BGR2GRAY, COLOR_BGR2RGB, inRange, kmeans,
    TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, KMEANS_RANDOM_CENTERS
)
from numpy import dstack, array, uint8, float32
import matplotlib.pyplot as plt

def threshold(image, th):
    return [[0 if cell < th else 255 for cell in row] for row in image]

img = imread('../resource/fight_club.jpg')

criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

img_gray = cvtColor(img, COLOR_BGR2GRAY)
img_rgb = cvtColor(img, COLOR_BGR2RGB)

Z = img_rgb.reshape((-1, 3))

ret, label, center = kmeans(float32(Z), 8, None, criteria, 8, KMEANS_RANDOM_CENTERS)

center = uint8(center)
res = center[label.flatten()]
img_kmean = res.reshape((img_rgb.shape))


img_rgb_threshold = inRange(img_rgb, array([10, 0, 5]), array([100, 100, 100]))

plt.subplot(231)
plt.title('Original image')
plt.imshow(img_rgb)

plt.subplot(232)
plt.title('K-means clustering')
plt.imshow(img_kmean)

plt.subplot(233)
plt.title('Thresholded color image')
plt.imshow(img_rgb_threshold, cmap='gray')

plt.subplot(223)
plt.title('Simple thresholding (gray)')
plt.imshow(threshold(img_gray, 100), cmap='gray')

plt.subplot(224)
plt.title('Adaptive threshold (gray)')
plt.imshow(adaptiveThreshold(img_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 12), cmap='gray')

plt.show()
