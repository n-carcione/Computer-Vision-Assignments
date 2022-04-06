'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

# Load hand picked points from im1 and separate into x and y coordinates
templeCoords = np.load('../data/templeCoords.npz')
x1 = templeCoords['x1']
y1 = templeCoords['y1']

# Load in camera intrinsics
intrinsics = np.load('../data/intrinsics.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
M = 640

# Load in F matrix and find E matrix
q2_1 = np.load('../results/q2_1.npz')
F = q2_1['F']
E = sub.essentialMatrix(F, K1, K2)

# Find the set of x2, y2 using the given points and F
# Create the pts1 and pts2 vectors out of the separate x and y values
x2 = np.zeros((len(x1),1)).astype(int)
y2 = np.zeros((len(y1),1)).astype(int)
for pt in np.arange(len(x1)):
    x2[pt,0], y2[pt,0] = sub.epipolarCorrespondence(im1, im2, F, x1[pt,0], y1[pt,0])
pts1 = np.hstack((x1,y1))
pts2 = np.hstack((x2,y2))

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
#Separate the x, y, and z coordinates
x_3d = P[:,0]
y_3d = P[:,1]
z_3d = P[:,2]

#Plot the 3D coordinates for inspection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_3d,y_3d,z_3d,s=1.5)

np.savez('../results/q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)