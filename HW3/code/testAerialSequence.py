import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.75, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

for frame in np.arange(1,seq.shape[2]):
    print(frame)
    mask = SubtractDominantMotion(seq[:,:,frame-1], seq[:,:,frame], threshold, num_iters, tolerance)
    
    if (frame == 30 or frame == 60 or frame == 90 or frame == 120): 
        plt.figure()
        plt.imshow(seq[:,:,frame-1], cmap='gray')
        for r in range(mask.shape[0]-1):
            for c in range(mask.shape[1]-1):
                if mask[r,c]:
                    plt.scatter(c, r, c = 'b', alpha=0.5, s=1)
        plt.show()