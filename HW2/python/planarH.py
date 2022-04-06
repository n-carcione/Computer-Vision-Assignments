import numpy as np
import cv2
import random


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    A = np.zeros((len(x1)*2,9))
    for i in np.arange(len(x1)):
        A[2*i,:] = [x2[i,0], x2[i,1], 1, 0, 0, 0, -x2[i,0]*x1[i,0], -x2[i,1]*x1[i,0], -x1[i,0]]
        A[2*i+1,:] = [0, 0, 0, x2[i,0], x2[i,1], 1, -x2[i,0]*x1[i,1], -x2[i,1]*x1[i,1], -x1[i,1]]

    u, s, vh = np.linalg.svd(A)
    h = vh[8,:]
    
    H2to1 = h.reshape(3,3)
    
    return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
    x_cent_1 = np.sum(x1[:,0]) / len(x1)
    y_cent_1 = np.sum(x1[:,1]) / len(x1)
    x_cent_2 = np.sum(x2[:,0]) / len(x2)
    y_cent_2 = np.sum(x2[:,1]) / len(x2)

	#Shift the origin of the points to the centroid
    x1[:,0] = x1[:,0] - x_cent_1
    x1[:,1] = x1[:,1] - y_cent_1
    x2[:,0] = x2[:,0] - x_cent_2
    x2[:,1] = x2[:,1] - y_cent_2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    dist1 = np.sqrt(np.power(x1[:,0:1],2) + np.power(x1[:,1:2],2))
    dist2 = np.sqrt(np.power(x2[:,0:1],2) + np.power(x2[:,1:2],2))
    x1 = x1 * np.sqrt(2) / max(dist1)[0]
    x2 = x2 * np.sqrt(2) / max(dist2)[0]

	#Similarity transform 1
    T1 = ( np.sqrt(2) / max(dist1)[0] ) * np.array(([1,0,-x_cent_1],[0,1,-y_cent_1],[0,0,max(dist1)[0]/np.sqrt(2)]))

	#Similarity transform 2
    T2 = ( np.sqrt(2) / max(dist2)[0] ) * np.array(([1,0,-x_cent_2],[0,1,-y_cent_2],[0,0,max(dist2)[0]/np.sqrt(2)]))

	#Compute homography
    H2to1 = computeH(x1,x2)

	#Denormalization
    H2to1 = np.matmul(np.matmul(np.linalg.inv(T1), H2to1), T2)

    return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    num_pts_used = 4
    comp_locs = np.zeros((3, len(locs2)))
    rand_pts = np.zeros((num_pts_used)).astype(int)
    num_inliers = -1
    for i in range(max_iters):
        #get the specified number of random points (all points unique)
        rand_pts = random.sample(range(len(locs1)),num_pts_used)
        #compute the homography using these random points
        H = computeH_norm(locs1[rand_pts],locs2[rand_pts])
        #code would throw a fit if I didn't redefine comp_locs here
        comp_locs = np.zeros((3, len(locs2)))
        #compute the resulting locations of H*x2
        for k in np.arange(len(locs2)):
            comp_locs[:,k] = np.matmul(H, np.array(([locs2[k,0]],[locs2[k,1]],[1]))).reshape(3).transpose()
        #convert to nonhomogeneous coordinates, transpose to match shape of x1,
        #and remove the final column (homogeneous part)
        comp_locs[0,:] = comp_locs[0,:]/comp_locs[2,:]
        comp_locs[1,:] = comp_locs[1,:]/comp_locs[2,:]
        comp_locs = np.transpose(comp_locs)
        comp_locs = comp_locs[:,[0,1]]
        #find the x and y distances between the computed locations and the actual locs1
        dists = locs1 - comp_locs
        #find the total geometric distance between the computed and actual points
        dists = np.sqrt(np.power(dists[:,0:1],2) + np.power(dists[:,1:2],2))
        #anywhere the distance is within tolerance, place a 1
        #anywhere it is not, place a 0
        ins = np.where(dists < inlier_tol,1,0)
        #if this homography results in the most inliers, update it as the best
        if (np.sum(ins) > num_inliers):
            num_inliers = np.sum(ins)
            inliers = ins
            bestH2to1 = H
    
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template
    mask = np.zeros((template.shape[0],template.shape[1],3)).astype(np.uint8) + 255

	#Warp mask by appropriate homography
    mask_warped = cv2.warpPerspective(mask,H2to1,(img.shape[1],img.shape[0]))

	#Warp template by appropriate homography
    template_warped = cv2.warpPerspective(template,H2to1,(img.shape[1],img.shape[0]))

	#Use mask to combine the warped template and the image
    composite_img = np.where(mask_warped != 0, template_warped, img)
	
    return composite_img


