import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

#Load data and initial given rectangle    
seq = np.load("../data/girlseq.npy")
# seq = seq/255
rect = [280, 152, 330, 318]     #original
# rect = [295, 162, 345, 328]     #makes output look the same as examples

#Create array for storing rectangles found from LK tracker
rects_list = np.zeros((seq.shape[2],4))
rects_list[0,:] = rect
p = np.zeros(2)

#Track the girl throughout the frames
for frame in np.arange(1,seq.shape[2]):
    p = LucasKanade(seq[:,:,frame-1], seq[:,:,frame], rects_list[frame-1,:].astype(int), threshold, int(num_iters))
    rects_list[frame,0] = rects_list[frame-1,0] + p[0]
    rects_list[frame,1] = rects_list[frame-1,1] + p[1]
    rects_list[frame,2] = rects_list[frame-1,2] + p[0]
    rects_list[frame,3] = rects_list[frame-1,3] + p[1]

#Plot the performance at the specified frames
frames_plt = [0, 19, 39, 59, 79]
for frm in frames_plt:
    figure, ax = plt.subplots(1)
    x1, y1, x2, y2 = rects_list[frm,0], rects_list[frm,1], rects_list[frm,2], rects_list[frm,3]
    img_rect = patches.Rectangle((x1,y1),x2+1-x1,y2+1-y1, edgecolor='r', facecolor="none")
    ax.imshow(seq[:,:,frm],cmap='gray')
    ax.add_patch(img_rect)

#Save rectangle corner coordinates    
np.save('../results/girlseqrects.npy',rects_list)