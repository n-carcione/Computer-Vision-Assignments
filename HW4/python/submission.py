"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import random
import scipy

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    #Normalize point coordinates by dividing by scalar M (largest image dim.)
    pts1_norm = pts1 / M
    pts2_norm = pts2 / M
    #Blank skeleton for U
    num_pts = len(pts1)
    U = np.zeros((num_pts,9))
    
    #Construct U from the point correspondences
    for corr in np.arange(num_pts):
        x, y = pts1_norm[corr,0], pts1_norm[corr,1]
        xp, yp = pts2_norm[corr,0], pts2_norm[corr,1]
        U[corr,:] = [x*xp, y*xp, xp, x*yp, y*yp, yp, x, y, 1]
    
    #Run SVD on U to find eigenvector that corresponds to lowest eigenvalue
    #Eigenvectors are the rows of vh
    #Should yield same result as U.T @ U
    (u, s, vh) = np.linalg.svd(U)
    #Reshape lowest e-value eigenvector to a 3x3 matrix
    F = np.reshape(vh[-1,:], (3,3))    
    #Make that matrix singular
    F = util._singularize(F)
    #Refine the matrix
    F = util.refineF(F, pts1_norm, pts2_norm)
    #Un-normalize/unscale F
    T = np.array(([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]]))
    F = T.T @ F @ T
    # np.savez('../results/q2_1.npz',F=F, M=M)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # F = inv(K1).T @ E @ inv(K2)
    # K1.T @ F @ K2 = E
    E = K1.T @ F @ K2                       # K1 and K2 may need to be swapped
    # np.savez('../results/q3_1.npz',E=E)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    num_pts = len(pts1)
    P = np.zeros((num_pts,3))
    err = 0
    for i in np.arange(num_pts):
        x1, y1 = pts1[i,0], pts1[i,1]
        x2, y2 = pts2[i,0], pts2[i,1]
        A = np.array(([ [C1[2,0]*y1-C1[1,0], C1[2,1]*y1-C1[1,1], C1[2,2]*y1-C1[1,2], C1[2,3]*y1-C1[1,3]],
                        [C1[0,0]-C1[2,0]*x1, C1[0,1]-C1[2,1]*x1, C1[0,2]-C1[2,2]*x1, C1[0,3]-C1[2,3]*x1],
                        [C2[2,0]*y2-C2[1,0], C2[2,1]*y2-C2[1,1], C2[2,2]*y2-C2[1,2], C2[2,3]*y2-C2[1,3]],
                        [C2[0,0]-C2[2,0]*x2, C2[0,1]-C2[2,1]*x2, C2[0,2]-C2[2,2]*x2, C2[0,3]-C2[2,3]*x2] ]))
        (U, S, VH) = np.linalg.svd(A)
        P_h = VH[-1,:]
        # print(P_h[3])
        P[i,:] = P_h[0:3] / P_h[3]
        # P[i,:] = P_h[0:3]
        proj1 = C1 @ (np.hstack((P[i,:],1)).T)
        proj1 = proj1[0:2] / proj1[2]
        # print(pts1[i,:])
        # print(proj1)
        proj2 = C2 @ (np.hstack((P[i,:],1)).T)
        proj2 = proj2[0:2] / proj2[2]
        # print(pts2[i,:])
        # print(proj2)
        err += (np.linalg.norm(pts1[i,:]-proj1) + np.linalg.norm(pts2[i,:]-proj2))
    
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Define the size of the window to be used later
    window_size = 15
    window_length = int((window_size - 1) / 2)
    
    # Create the homogeneous coordinates of the pixel from im1
    pt1 = np.array(([ [x1], [y1], [1] ]))
    # Find the epipolar line in im2 that corresponds to [x1,y1]
    l = F @ pt1
    a, b, c = l[0,0], l[1,0], l[2,0]
    
    # Get the intensity values of im1 within the window surround pt1
    # Window converted into a vector that contains intensity values of all 3
    # channels starting at top left of window, working right across columns,
    # then down rows.  The 3 channels are kept together for each pixel (i.e.,
    # 3 channel intensities for top left, then 3 channel intensities for next pixel)
    im1_window = im1[y1-window_length:y1+window_length+1, x1-window_length:x1+window_length+1]
    im1_window = im1_window.reshape((window_size**2 * 3, 1))
    
    x2_list = np.zeros((im2.shape[0],1))
    y2_list = np.zeros((im2.shape[0],1))
    dist_list = np.zeros((im2.shape[0],1))
    for row in np.arange(im2.shape[0]):
        # Since we know the epipolar lines are near vertical, go thru
        # each row and find the column of the pixel that lies on/nearest
        # to where the epipolar line would be
        y2i = row
        y2_list[row,0] = y2i
        x2i = round((-b*y2i - c) / a)
        x2_list[row,0] = x2i
        
        # Rows at very top and bottom are skipped to avoid indexing errors
        # These rows have their distances set very high to disqualify them from
        # being matches
        if y2i < window_length or y2i > im2.shape[0] - 1 - window_length:
            dist_list[row,0] = 1000000
            continue
        
        # Get a vector of the im2 window in the same format as the im1 window
        im2_window = im2[y2i-window_length:y2i+window_length+1,x2i-window_length:x2i+window_length+1]
        im2_window = im2_window.reshape((window_size**2 * 3, 1))
        
        # From the write up, since the two images differ by only a small amount
        # prioritize points in im2 that are near to the original point from 
        # im1.  Without this, results were very noisy and inaccurate. With this,
        # the results vastly improved
        dist_list[row,0] = np.linalg.norm(im1_window - im2_window)
        if abs(x2i - x1) > 30 or abs(y2i - y1) > 30:
            dist_list[row,0] += 100
    
    # Return the [x2,y2] coordinates that are most similar to the [x1,y1] window
    min_dist_ind = np.argmin(dist_list)
    x2 = int(x2_list[min_dist_ind,0])
    y2 = int(y2_list[min_dist_ind,0])
    
    return x2, y2

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.82):
    #use the minimum number of points to find F (8)
    num_pts_used = 8
    #initialize all of the inliers to 0/false
    inliers = np.zeros((len(pts1),1),dtype=bool)
    #initialize F to the identity
    F = np.identity(3)
    for i in range(nIters):
        #get the specified number of random points (all points unique)
        rand_pts = random.sample(range(len(pts1)),num_pts_used)
        #use the 8 randomly selected points to find F
        F_r = eightpoint(pts1[rand_pts], pts2[rand_pts], M)
        #set of blank inliers for this iteration
        ins = np.zeros((len(pts1),1),dtype=bool)
        #for each point correspondence, use the found F_r and the point in
        #image 1 to find the epipolar line in image 2
        #use the corresponding point in image 2 and the line coefficients to
        #find the perpendicular distance between the point and the line
        #since the point should be on the line, the value of this distance
        #is the error metric
        for k in np.arange(len(pts1)):
            pt1 = np.array(([ [pts1[k,0]], [pts1[k,1]], [1] ]))
            pt2 = np.array(([ [pts2[k,0]], [pts2[k,1]], [1] ]))
            l_2 = F_r @ pt1
            a, b, c = l_2[0,0], l_2[1,0], l_2[2,0]
            err = np.abs(a*pt2[0,0] + b*pt2[1,0] + c) / np.sqrt(a**2 + b**2)
            #if the error (distance) is less than the tolerance, the point is an inlier
            if err < tol:
                ins[k,0] = True
        #update the inliers and best F guess to be the one with the most inliers
        if (np.sum(ins) > np.sum(inliers)):
            inliers = ins
            F = F_r
    return F, inliers


'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    #Separate r into kx, ky, and kz to match notation in notes
    kx = r[0]
    ky = r[1]
    kz = r[2]
    #Construct skew-symmetric K matrix
    K = np.array(([ [0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0] ]))
    #theta is the magnitude of r
    theta = np.linalg.norm(r)
    
    #Calculate rotation matrix R using Rodrigues formula
    R = np.identity(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
    
    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    #Use R to find theta and omega using formulae given in class
    theta = np.arccos( (np.trace(R) - 1) / 2 )
    omega = np.array(([ [R[2,1]-R[1,2]], [R[0,2]-R[2,0]], [R[1,0]-R[0,1]] ])) / (2*np.sin(theta))
    #Find r from theta and omega
    r = theta * omega
    
    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenation of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(x, K1, M1, p1, K2, p2):
    #get w, r2, and t2 out of x
    w = x[0:len(x)-6]
    w = w.reshape((int(len(w)/3),3))
    r2 = x[len(x)-6:len(x)-3]
    t2 = x[len(x)-3:len(x)]
    t2 = t2.reshape((3,1))
    #find R2 from r2 and create M2
    R2 = rodrigues(r2)
    M2 = np.hstack((R2,t2))
    #calculate the camera centers
    C1 = K1 @ M1
    C2 = K2 @ M2
    
    residuals = np.zeros((4*len(w),1))
    for i in np.arange(len(w)):
        #find the projection of the 3D point onto camera 1
        rp1 = C1 @ (np.hstack((w[i,:],1)).T)
        rp1 = rp1[0:2] / rp1[2]
        #find the projection of the 3D point onto camera 2
        rp2 = C2 @ (np.hstack((w[i,:],1)).T)
        rp2 = rp2[0:2] / rp2[2]
        #find the error between the x and y coordinates of the 2 projections to
        #where the 2 points actually are
        err1 = p1[i,:] - rp1
        err2 = p2[i,:] - rp2
        #update the residuals vector
        residuals[4*i,0] = err1[0]
        residuals[4*i+1,0] = err1[1]
        residuals[4*i+2,0] = err2[0]
        residuals[4*i+3,0] = err2[1]
    
    return residuals.flatten()

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    #break M2 into R2 and t2 and convert R2 to r2
    R2 = M2_init[:,0:3]
    r2 = invRodrigues(R2)
    t2 = M2_init[:,3].reshape((3,1))
    #Reshape the 3D points into a single vector
    w = P_init.reshape((P_init.shape[0]*P_init.shape[1],1))
    #Construct the x vector for use in rodriguesResidual()
    x = np.vstack((w,r2,t2))

    #Optimize the values in x to minimize the residuals
    x_opt = scipy.optimize.leastsq(rodriguesResidual, x, args=(K1, M1, p1, K2, p2))
    #Separate out the optimized 3D points and M2 matrix
    P2 = x_opt[0][0:len(x_opt[0])-6]
    P2 = P2.reshape(int(len(P2)/3),3)
    r2_opt = x_opt[0][len(x_opt[0])-6:len(x_opt[0])-3]
    t2_opt = x_opt[0][len(x_opt[0])-3:len(x_opt[0])]
    t2_opt = t2_opt.reshape((3,1))
    R2_opt = rodrigues(r2_opt)
    M2 = np.hstack((R2_opt, t2_opt))
    
    return M2, P2
