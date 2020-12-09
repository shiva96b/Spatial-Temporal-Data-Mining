import cv2
from PIL import Image

# Read image2
image2 = cv2.imread("./slide_parallel_finale.png")
print(image2.shape)

# Read Image1 and crop the required region
image1 = cv2.imread("./k12-05m.tif")
image1 = image1[2000:2600, 2400:3000]
print(image1.shape)
cv2.imwrite('image1.png',image1)


# Blend Images
img = cv2.addWeighted(image2, 0.2, image1, 0.8, 0)
cv2.imwrite('image2.png',img)

# Show the image
cv2.imwrite('blend.png', img)
