# References
# https://www.tutorialspoint.com/dip/sobel_operator.htm
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Converting the image to grayscale
img = np.array(Image.open('einstein.jpg')).astype(np.uint8)
img = np.round(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


horizontal_sobel = np.zeros((img.shape[0], img.shape[1]))
vertical_sobel = np.zeros((img.shape[0], img.shape[1]))

for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):

        temp = (-1 * img[i-1][j-1]) + img[i-1][j + 1] + (-2 * img[i][j-1]) + (2 * img[i][j+1]) + (-1 * img[i+1][j-1]) + img[i+1][j+1]
        horizontal_sobel[i-1][j-1] = abs(temp)

        temp = (-1 * img[i-1][j-1]) + (-2 * img[i-1][j]) + (-1 * img[i-1][j+1]) + img[i+1][j-1] + (2 * img[i+1][j]) + img[i+1][j+1]
        vertical_sobel[i-1][j-1] = abs(temp)

plt.figure()
plt.imsave('einstein-sobel-horizontal.png', vertical_sobel, cmap='gray', format='png')
plt.imsave('einstein-sobel-vertical.png', horizontal_sobel, cmap='gray', format='png')
