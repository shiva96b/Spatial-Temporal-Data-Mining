# References
# https://scikit-learn.org/
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import mixture

np.random.seed(seed=5)

#Data Reading and Sampling
img = cv2.imread("ilk-3b-1024.tif")
img = np.array(img)
width, height, color = img.shape
image_array = np.reshape(img, (width * height, color))
sample_size = int(0.05 * len(image_array))
sample = np.copy(image_array)
np.random.shuffle(sample)
sample = sample[:sample_size]

plt.figure(1)
# K-Means Clustering
kmeans = KMeans(n_clusters = 5).fit(sample)
predict_kmeans = kmeans.predict(image_array)
predict_kmeans = predict_kmeans.reshape(height,width)
plt.imshow(predict_kmeans)
plt.imsave('kmeans.png', predict_kmeans, format='png')

plt.figure(2)
# Model-based Clustering (GMM)
gmm = mixture.GaussianMixture(n_components = 5).fit(sample)
predict_gmm = gmm.predict(image_array)
predict_gmm = predict_gmm.reshape(width,height)
plt.imshow(predict_gmm)
plt.imsave('gmm.png', predict_gmm, format='png')

plt.show()
