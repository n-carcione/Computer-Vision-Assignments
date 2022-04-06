# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 23:35:59 2021

@author: Nick Carcione
"""

import numpy as np
import submission as sub
import matplotlib.pyplot as plt
import helper

noisy = np.load('../data/some_corresp_noisy.npz')
pts1 = noisy['pts1']
pts2 = noisy['pts2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
M = 640

F_no_ransac = sub.eightpoint(pts1, pts2, M)
F_ransac, inliers = sub.ransacF(pts1, pts2, M)
# helper.displayEpipolarF(im1, im2, F_ransac)

E = sub.essentialMatrix(F_ransac, K1, K2)

ins = 0
pts1_ins = np.zeros((np.sum(inliers),2))
pts2_ins = np.zeros((np.sum(inliers),2))
for pt in np.arange(len(pts1)):
    if inliers[pt,0]:
        pts1_ins[ins,:] = pts1[pt,:]
        pts2_ins[ins,:] = pts2[pt,:]
        ins += 1

# Create camera matrix C1 = K1 [I | 0]
M1 = np.hstack((np.identity(3),np.zeros((3,1))))
C1 = K1 @ M1
# Get the 4 potential M2s based on E
M2s = helper.camera2(E)

#Triangulate all of the 3D points for all 4 potential M2s
num_pts = len(pts1_ins)
Ps = np.zeros((num_pts, 3, 4))
errs = np.zeros((4,1))
num_negs = np.zeros((4,1)).astype(int)
for i in np.arange(4):
    C2 = K2 @ M2s[:,:,i]
    [Ps[:,:,i], errs[i,0]] = sub.triangulate(C1, pts1_ins, C2, pts2_ins)
    for k in np.arange(Ps.shape[0]):
        if Ps[k,2,i] < 0:
            num_negs[i] += 1

#Select the best M2 as the one with no negative z-values
correct = np.argmin(num_negs)
M2 = M2s[:,:,correct]
C2 = K2 @ M2
P = Ps[:,:,correct]

M2_bund, P_bund = sub.bundleAdjustment(K1, M1, pts1_ins, K2, M2, pts2_ins, P)

x_3d = P_bund[:,0]
y_3d = P_bund[:,1]
z_3d = P_bund[:,2]

#Plot the 3D coordinates for inspection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_3d,y_3d,z_3d,s=1.5)