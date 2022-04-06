'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

# Load in data and other givens
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
M = 640

# Get F and E
F = sub.eightpoint(pts1, pts2, M)
E = sub.essentialMatrix(F, K1, K2)

# Create camera matrix C1 = K1 [I | 0]
M1 = np.hstack((np.identity(3),np.zeros((3,1))))
C1 = K1 @ M1
# Get the 4 potential M2s based on E
M2s = helper.camera2(E)

#Triangulate all of the 3D points for all 4 potential M2s
num_pts = len(pts1)
Ps = np.zeros((num_pts, 3, 4))
errs = np.zeros((4,1))
num_negs = np.zeros((4,1)).astype(int)
for i in np.arange(4):
    C2 = K2 @ M2s[:,:,i]
    [Ps[:,:,i], errs[i,0]] = sub.triangulate(C1, pts1, C2, pts2)
    for k in np.arange(Ps.shape[0]):
        if Ps[k,2,i] < 0:
            num_negs[i] += 1

#Select the best M2 as the one with no negative z-values
correct = np.argmin(num_negs)
M2 = M2s[:,:,correct]
C2 = K2 @ M2
P = Ps[:,:,correct]
np.savez('../results/q3_3.npz', M2=M2, C2=C2, P=P)
    