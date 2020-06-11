from cv2 import imread, cvtColor, COLOR_BGR2RGB, Canny, Laplacian, Sobel, CV_8U, medianBlur, GaussianBlur, blur
from numpy import array
import matplotlib.pyplot as plt

img = imread('../resource/inception.jpg')
img_rgb = cvtColor(img, COLOR_BGR2RGB)
# img_rgb_blur = blur(img_rgb, (3, 3))
img_rgb_blur = GaussianBlur(img_rgb, (3, 3), 8)  # Gaussian
# img_rgb_blur = medianBlur(img_rgb, 3)  # median

img_can = Canny(img_rgb, 180, 220, apertureSize=3)
img_can_blur = Canny(img_rgb_blur, 180, 220, apertureSize=3)
img_lap = Laplacian(img_rgb, CV_8U)
img_sob = Sobel(img_rgb, CV_8U, 1, 1)

plt.subplot(231)
plt.title('Original image')
plt.imshow(img_rgb)

plt.subplot(232)
plt.title('Canny')
plt.imshow(img_can)

plt.subplot(233)
plt.title('Canny (with blur)')
plt.imshow(img_can_blur)

plt.subplot(223)
plt.title('Laplacian')
plt.imshow(img_lap)

plt.subplot(224)
plt.title('Sobel')
plt.imshow(img_sob)

plt.show()

