from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize
import matplotlib.pyplot as plt

img = imread('../resource/target.png')
img = cvtColor(img, COLOR_BGR2RGB)  # OpenCV uses BGR so we have to convert for it to look normal

resized_img = resize(img, (512, 512))

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(resized_img)

plt.show()

