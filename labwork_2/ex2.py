from cv2 import imread, cvtColor, COLOR_BGR2RGB, Canny, Laplacian, Sobel, CV_32F
from numpy import array
import matplotlib.pyplot as plt

img = imread('../resource/target.png')
img_rgb = cvtColor(img, COLOR_BGR2RGB)

img_can = Canny(img_rgb, 10, 100)
img_lap = Laplacian(img_rgb, CV_32F)
img_sob = Sobel(img_rgb, CV_32F, 1, 1)

plt.subplot(221)
plt.title('Original image')
plt.imshow(img_rgb)

plt.subplot(222)
plt.title('Canny')
plt.imshow(img_can)

plt.subplot(223)
plt.title('Laplacian')
plt.imshow(img_lap)

plt.subplot(224)
plt.title('Sobel')
plt.imshow(img_sob)

plt.show()

