from cv2 import imread, cvtColor, COLOR_BGR2RGB, blur, GaussianBlur, medianBlur
import matplotlib.pyplot as plt

img = imread('../resource/target.png')
img_rgb = cvtColor(img, COLOR_BGR2RGB)

img_avg = blur(img_rgb, (10, 10))
img_gss = blur(img_rgb, (10, 10), 10)
img_med = blur(img_rgb, (10, 10))

plt.subplot(221)
plt.title('Original image')
plt.imshow(img_rgb)

plt.subplot(222)
plt.title('Average blur')
plt.imshow(img_avg)

plt.subplot(223)
plt.title('Gaussian blur')
plt.imshow(img_gss)

plt.subplot(224)
plt.title('Median blur')
plt.imshow(img_med)

plt.show()

