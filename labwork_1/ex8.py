from cv2 import imread, cvtColor, COLOR_BGR2RGB, Laplacian, Sobel, CV_32F
import matplotlib.pyplot as plt

img = imread('../resource/target.png')
img_rgb = cvtColor(img, COLOR_BGR2RGB)

img_lap = Laplacian(img_rgb, CV_32F)
img_sob = Sobel(img_rgb, CV_32F, 1, 1)

plt.subplot(131)
plt.title('Original image')
plt.imshow(img_rgb)

plt.subplot(132)
plt.title('Laplacian')
plt.imshow(img_lap)

plt.subplot(133)
plt.title('Sobel')
plt.imshow(img_sob)

plt.show()

