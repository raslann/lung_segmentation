import numpy as np
import sys
import os
import dicom
import matplotlib
import matplotlib.pyplot as plt
import argparse

from skimage import morphology
from skimage import measure

from sklearn.cluster import KMeans

def find_threshold(image):
    kmeans = KMeans(n_clusters=2).fit(image.reshape(-1,1))
    c = kmeans.cluster_centers_
    return 0.5*(c[0]+c[1])

parser = argparse.ArgumentParser(description='lung_segmentation')
parser.add_argument('--input', default='default.dcm', help='any .dcm file')


input = 'default.dcm'
scan = dicom.read_file(input)
image = scan.pixel_array
image[image == -2000] = 0
intercept = scan.RescaleIntercept
slope = scan.RescaleSlope
image += np.int16(intercept)
image = np.array(image, dtype=np.int16)


threshold = find_threshold(image)
binary_image = (image > threshold).astype(int) 


image_o = morphology.binary_opening(binary_image,morphology.diamond(7))


labels = measure.label(image_o+1,neighbors=4)  # add one to image_o because '0' is considered to be 'background'
label_vals = np.unique(labels)

fig = plt.figure(figsize=(7,7))
ax0 = fig.add_subplot(221); ax0.imshow(labels); ax0.set_title('Segmentation')
ax1 = fig.add_subplot(222); ax1.imshow(labels==2, cmap='gray'); ax1.set_title('Wall')

plt.savefig('output.png')


