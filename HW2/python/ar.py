import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
import planarH
import loadVid
# from multiprocessing import Pool
# import multiprocessing
from opts import get_opts

#Write script for Q3.1
def frameByFrame(cv_cover, panda_frame, book_frame, opts):
    #resize the panda frame to be the same size as cv_cover
    panda_frame = cv2.resize(panda_frame,(cv_cover.shape[1], cv_cover.shape[0]))
    #get the matches between cv_cover and the boook frame
    matches, locs1, locs2 = matchPics(book_frame, cv_cover, opts)
    #reduce locs1 and locs2 to just the ordered matches
    locs1 = locs1[matches[:,0]]
    locs2 = locs2[matches[:,1]]
    #swap locs1 and locs2 columns from (y,x) to (x,y)
    locs1[:, [1,0]] = locs1[:,[0,1]]
    locs2[:, [1,0]] = locs2[:,[0,1]]
    #compute the homography between cv_cover and book frame
    H, inliers = planarH.computeH_ransac(locs1, locs2, opts)
    #use that homography to overlay panda frame onto book cover in video frame
    result = planarH.compositeH(H,panda_frame,book_frame)
    return result

#load in the videos
#note: panda and book in the form [frame #, rows, cols, 3]
#note: panda has fewer frames/is shorter than book
panda = loadVid.loadVid('../data/ar_source.mov')
book = loadVid.loadVid('../data/book.mov')

#load in cv_cover
cv_cover = cv2.imread('../data/cv_cover.jpg')

#compute aspect ratio of cv_cover
#asp is the ratio of #cols to #rows
asp = cv_cover.shape[1] / cv_cover.shape[0]

#crop panda frames to be same aspect ratio as cv_cover
num_cols_keep = round((315-44+1) * asp)-1
left_keep = int(panda.shape[2] / 2) - 1 - int(num_cols_keep / 2) + 1
right_keep = int(panda.shape[2] / 2) + int(num_cols_keep / 2) - 1
panda_crop = panda[:,44:315,left_keep:right_keep+1,:]

#for each frame of both videos, overlay the cv textbook cover with the appropriate
#frame from the Kung Fu Panda video
#note: this is done for each fram in the panda video (some book frames are left off)
opts = get_opts()
frames = np.zeros((panda.shape[0],book.shape[1],book.shape[2],3)).astype(np.uint8)
for i in range(panda.shape[0]):
    frames[i] = frameByFrame(cv_cover, panda_crop[i], book[i], opts)
    print(i)

#write all of the frames to a video
out = cv2.VideoWriter('.../result/ar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (book.shape[2],book.shape[1]))
for i in range(panda.shape[0]):
    out.write(frames[i])
out.release()