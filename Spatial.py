import cv2
import numpy as np
from joblib import Parallel, delayed
from numpy import asarray
from numpy.linalg import slogdet, pinv
from math import log
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import math
import multiprocessing
import datetime

# Multiprocessor Threading
num_cores = multiprocessing.cpu_count()
print("Number of Cores: ", num_cores)

# Finding KL Divergence between 2 Gaussians
def KL_divergance(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    try:
        sum = 0
        for i in range(len(a)):
            if a[i]!=0 and b[i]!=0:
                sum += a[i] * np.log(a[i] / b[i])
        # sum = np.sum(np.where(b != 0 and a!=0, a * np.log(a / b), 0))
        return sum
    except:
        return 0


# Reference Scipydocs.scipy.org
# Finding the gaussian distribution of the window

def gaussian(input, u, sigma):
    n = len(input)
    sdet = slogdet(sigma)
    diff = input - u
    p = (np.matmul(np.matmul(diff[None, :], pinv(sigma)), diff[None, :].T))
    log_likelihood = (-n / 2 * log(2 * np.pi)) + (-0.5 * sdet[1]) + (0.5 * p)
    return log_likelihood


# Reading the 2 images
img1 = cv2.imread('k02-05m.tif',0)
img2 = cv2.imread('k12-05m.tif',0)

# Converting the pixels to manipulatable information
data1 = asarray(img1)
data2 = asarray(img2)

# The dimensions of both the images are the same
im_height, im_width = img1.shape

# print(img1.shape)
# print(img2.shape)

# Setting the window size
window_height, window_width = 25,25
output_height = im_height - window_height
output_width = im_width - window_width

# Result Array
distance_list = []

# Start time Notation
then = datetime.datetime.now()
print(then.strftime("%Y-%m-%d %H:%M:%S"))

# Sliding Window Algorithm:
# Finding the KL Divergence for each window between Gaussian of the 2 images
def image_processing(i,j):
    # print(i,j)
    try:
        # Old Image
        mean1, std1 = cv2.meanStdDev(data1[i:i + window_height, j:j + window_width])
        mean1 = float(mean1)
        cov1 = np.cov(data1[i:i + window_height, j:j + window_width], rowvar=False)
        y1 = multivariate_normal.pdf(data1[i:i + window_height, j:j + window_width].flatten(), mean1, cov=std1)

        # New Image
        mean2, std2 = cv2.meanStdDev(data2[i:i + window_height, j:j + window_width])
        mean2 = float(mean2)
        cov2 = np.cov(data2[i:i + window_height, j:j + window_width])
        y2 = multivariate_normal.pdf(data2[i:i + window_height, j:j + window_width].flatten(), mean2, cov=std2)

        # KL Divergence
        distance1 = KL_divergance(y1, y2)
        distance2 = KL_divergance(y2, y1)
        if distance1 == math.inf or distance1 == math.inf:
            return 0
        else:
            print((distance1 + distance2)/2)
            return (distance1 + distance2)/2

    except:
        return 0

def grid_based_approach():
    distance_list = []
    for i in range(3000,3020):
        for j in range(3000,3020):
            distance = image_processing(i,j)
            distance_list.append([distance])

    distance_list = np.array(distance_list)

    r,c = 0,0
    #Grid bassed approach
    for i in range(100,100+window_height*100,window_height):
        r += 1
        for j in range(100,100+window_width*100,window_width):
            c += 1
            try:
                mean1,std1 = cv2.meanStdDev(data1[i:i+window_height,j:j+window_width])
                mean1 = float(mean1)
                cov1 = np.cov(data1[i:i+window_height,j:j+window_width], rowvar=False)
                y1 = multivariate_normal.pdf(data1[i:i+window_height,j:j+window_width].flatten(),mean1,cov = std1)

                mean2,std2 = cv2.meanStdDev(data2[i:i+window_height,j:j+window_width])
                mean2 = float(mean2)
                cov2 = np.cov(data2[i:i+window_height,j:j+window_width])
                y2 = multivariate_normal.pdf(data2[i:i+window_height,j:j+window_width].flatten(),mean2,cov = std2)

                distance1 = KL_divergance(y1,y2)
                distance2 = KL_divergance(y2,y1)
                if (distance1 + distance2) == math.inf:
                    distance_list.append([0])
                else:
                    distance_list.append([(distance1 + distance2)/2])

            except:
                distance_list.append([0])
    return distance_list

# Parallelization
print("No. of Cores Used: ", 4)
x = Parallel(n_jobs=num_cores)(delayed(image_processing)(i,j)
                                    for i in range(3000,3010) for j in range(3000,3010))

# Converitng to numpy array for easier and faster processing
distance_list = [[i] for i in x]
distance_list = np.array(distance_list)

# To call the grid based approach:
# distance_list = grid_based_approach()
# distance_list = np.array(distance_list)

# End time Notation
now = datetime.datetime.now()
print(then.strftime("%Y-%m-%d %H:%M:%S"))
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Model-based Clustering (GMM)
# EM Clustering (no. of clusters = 4) of the distances
plt.figure(1)
gmm = GaussianMixture(n_components=4).fit(distance_list)
labels = gmm.predict(distance_list)

# Generating the Change Map
predict_gmm = labels.reshape(10,10)
plt.imshow(predict_gmm)
plt.imsave('slide_parallel_finale.png', predict_gmm, format='png')

# Calculating algorithm run time
duration = now - then
duration_in_s = duration.total_seconds()
print("Duration: ", duration_in_s)
