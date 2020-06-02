from cv2 import imread
from numpy import dstack
import matplotlib.pyplot as plt

print('Input the parameters for brightening')
a = int(input())
b = int(input())

def brighten(image):
    return [[a * cell + b for cell in row] for row in image]

img = imread('../resource/target.png')
img_b = img[:, :, 0]
img_g = img[:, :, 1]
img_r = img[:, :, 2]

img_b = brighten(img_b)
img_g = brighten(img_g)
img_r = brighten(img_r)

img_bright = dstack((img_r, img_g, img_b))

plt.imshow(img_bright)

plt.show()
