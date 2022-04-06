import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
import planarH


#Write script for Q2.2.4
opts = get_opts()
#1.  Reads cv_cover.jpg, cv_desk.png, and hp_cover.jpg
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
#4. Resize hp_cover to be same size as cv_cover
hp_cover = cv2.resize(hp_cover,(cv_cover.shape[1], cv_cover.shape[0]))

#2. Computes a homography automatically using MatchPics and computeH_ransac
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
locs1 = locs1[matches[:,0]]
locs2 = locs2[matches[:,1]]
locs1[:, [1,0]] = locs1[:,[0,1]]
locs2[:, [1,0]] = locs2[:,[0,1]]
H, inliers = planarH.computeH_ransac(locs1, locs2, opts)

#3. Uses the computed homography to warp hp_cover.jpg to the dimensions of the
#   cv_desk.png image using the OpenCV function cv2.warpPerspective function
# hp_desk = cv2.warpPerspective(hp_cover,H,(cv_desk.shape[1],cv_desk.shape[0]))
# test = cv2.warpPerspective(cv_cover,H,(cv_desk.shape[1],cv_desk.shape[0]))
# cv2.imshow('test',hp_desk)
# cv2.imshow('test2',test)

#5. overlay hp_cover onto textbook cover in cv_desk
result = planarH.compositeH(H,hp_cover,cv_desk)
#display result
cv2.imshow('final',result)
# cv2.imwrite('Q_2_2_4.png', result)