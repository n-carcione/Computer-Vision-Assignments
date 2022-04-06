import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from scipy.interpolate import RectBivariateSpline

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
#Load girl sequence, given initial rectangle, and rectangles from base case
seq = np.load("../data/girlseq.npy")
old_rects = np.load("../results/girlseqrects.npy")
rect = [280, 152, 330, 318]

#Create an array for storing all the rectangles
rects_list = np.zeros((seq.shape[2],4))
rects_list[0,:] = rect

#Since the first template T doesn't change, compute it here one time only
rows_img = seq.shape[0]
cols_img = seq.shape[1]
x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
rows_rect = y2-y1
cols_rect = x2-x1
x = np.arange(0, cols_img)
y = np.arange(0, rows_img)
temp_cols, temp_rows = np.meshgrid(np.linspace(x1, x2, cols_rect), np.linspace(y1, y2, rows_rect))
spline = RectBivariateSpline(y, x, seq[:,:,0])
T = spline.ev(temp_rows, temp_cols)

#Track the car throughout the frames with template correction
for frame in np.arange(1,seq.shape[2]):
    It = seq[:,:,frame-1]
    It1 = seq[:,:,frame]
    
    p1 = LucasKanade(It, It1, rects_list[frame-1,:].astype(int), threshold, int(num_iters))
    rects_list[frame,0] = rects_list[frame-1,0] + p1[0]
    rects_list[frame,1] = rects_list[frame-1,1] + p1[1]
    rects_list[frame,2] = rects_list[frame-1,2] + p1[0]
    rects_list[frame,3] = rects_list[frame-1,3] + p1[1]
    
    p2 = LucasKanade(T, It1, rects_list[frame,:].astype(int), threshold, int(num_iters))
    rects_list[frame,0] = rects_list[frame,0] + p2[0]
    rects_list[frame,1] = rects_list[frame,1] + p2[1]
    rects_list[frame,2] = rects_list[frame,2] + p2[0]
    rects_list[frame,3] = rects_list[frame,3] + p2[1]


#Plot the performance at the specified frames
frames_plt = [0, 19, 39, 59, 79]
for frm in frames_plt:
    figure, ax = plt.subplots(1)
    x1, y1, x2, y2 = rects_list[frm,0], rects_list[frm,1], rects_list[frm,2], rects_list[frm,3]
    x1o, y1o, x2o, y2o = old_rects[frm,0], old_rects[frm,1], old_rects[frm,2], old_rects[frm,3]
    img_rect = patches.Rectangle((x1,y1),x2+1-x1,y2+1-y1, edgecolor='r', facecolor="none")
    img_old_rect = patches.Rectangle((x1o,y1o),x2o+1-x1o,y2o+1-y1o, edgecolor='b', facecolor="none")
    ax.imshow(seq[:,:,frm],cmap='gray')
    ax.add_patch(img_old_rect)
    ax.add_patch(img_rect)

#Save rectangle corner coordinates    
np.save('../results/girlseqrects-wcrt.npy',rects_list)