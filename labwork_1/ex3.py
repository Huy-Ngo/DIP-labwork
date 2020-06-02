from cv2 import imread
from numpy import dstack
import matplotlib.pyplot as plt

img = imread('../resource/target.png')
img_b = img[:, :, 0]
img_g = img[:, :, 1]
img_r = img[:, :, 2]

img_gray = img_b * 0.0722 + img_g * 0.7152 + img_r * 0.2126

plt.imshow(img_gray, cmap='gray')

plt.show()
