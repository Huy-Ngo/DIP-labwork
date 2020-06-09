from cv2 import imread, cvtColor, COLOR_BGR2RGB, HoughLines, Canny
import matplotlib.pyplot as plt
from numpy import pi, cos, sin
import matplotlib.lines as mlines

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='red')
    ax.add_line(l)
    return l

img = imread('../resource/inception.jpg')
img_rgb = cvtColor(img, COLOR_BGR2RGB)

edges = Canny(img_rgb, 180, 220, apertureSize=3)

lines = HoughLines(edges, 1, pi/180, 235)

plt.subplot(121)
plt.imshow(edges)

plt.subplot(122)
plt.imshow(img_rgb)
# print(lines)
for line in lines:
    for rho, theta in line:
        a = cos(theta)
        b = sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        newline([x1, y1], [x2, y2])
        # print(x1, x2, y1, y2)

plt.show()

