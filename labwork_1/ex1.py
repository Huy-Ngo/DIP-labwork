from cv2 import imread, cvtColor, COLOR_BGR2RGB
import matplotlib.pyplot as plt

img = imread('../resource/target.png')
img_rgb = cvtColor(img, COLOR_BGR2RGB)  # OpenCV uses BGR so we have to convert for it to look normal

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(img_rgb)

plt.show()

