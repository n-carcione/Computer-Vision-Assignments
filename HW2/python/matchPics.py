import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
    
    #Convert Images to GrayScale 
    im1 = skimage.color.rgb2gray(I1)
    im2 = skimage.color.rgb2gray(I2)

    #Detect Features in Both Images
    locs1 = corner_detection(im1, sigma)
    locs2 = corner_detection(im2, sigma)
	
    #Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(im1, locs1)
    desc2, locs2 = computeBrief(im2, locs2)

    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
