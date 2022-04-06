import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from matplotlib import pylab
from matchPics import matchPics
from opts import get_opts

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')

counts = []
bins = []
for i in range(36):
 	#Rotate Image
    rotated = scipy.ndimage.rotate(img, i*10)
 	
 	#Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(img, rotated, opts)

 	#Update histogram
    counts = np.append(counts, len(matches))
    bins = np.append(bins, i*10)


#Display histogram
bins = bins.astype(int)
counts = counts.astype(int).tolist()
fig = plt.figure()
plt.bar(bins,counts,width=10,edgecolor='black')
pylab.title('BRIEF Performance with Rotations')
pylab.xlabel('Rotation [deg]')
pylab.ylabel('# of Matches')
plt.show()
