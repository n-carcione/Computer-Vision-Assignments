import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here    
    
    p = np.zeros(6).reshape((1,6))
    
    #x and y values for It corners
    rows = It.shape[0]
    cols = It.shape[1]
    
    #Compute x- and y-direction gradients of original It1
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    It_spline = RectBivariateSpline(y, x, It)
    spline_It1 = RectBivariateSpline(y, x, It1)
    cc, rr = np.meshgrid(np.linspace(0,cols-1,cols), np.linspace(0,rows-1,rows))
    # Itx = It1_spline.ev(rr,cc, dy=1)
    # Ity = It1_spline.ev(rr,cc, dx=1)
    Its = It_spline.ev(rr,cc)
    # I = np.vstack((Itx.ravel(), Ity.ravel())).T
    
    coords = np.vstack((cc.ravel(),rr.ravel(),np.ones((rows*cols))))
    mask = np.ones(It1.shape, dtype=int)
    mask_spline = RectBivariateSpline(y, x, mask)
    
    #Do the following for num_iters
    for iter in np.arange(num_iters):
        
        M1 = np.array([[1.0+p[0,0], p[0,2], p[0,4]], [p[0,1], 1.0+p[0,3], p[0,5]], [0, 0, 1]])
        
        coords_w = M1 @ coords
        It1_warped = spline_It1.ev(coords_w[1,:],coords_w[0,:])
        It1_warped = It1_warped.reshape(It.shape)
        mask_warped = mask_spline.ev(coords_w[1,:],coords_w[0,:])
        mask_warped = mask_warped.reshape(It.shape)
        It1x_warped = spline_It1.ev(coords_w[1,:],coords_w[0,:], dy=1)
        It1y_warped = spline_It1.ev(coords_w[1,:],coords_w[0,:], dx=1)
        It_kept = Its*mask_warped
        
        error_img = It_kept - It1_warped
        # cv2.imshow("Error", error_img)        
        error_img = error_img.reshape(-1,1)
        
        I = np.vstack((It1x_warped.ravel(), It1y_warped.ravel())).T
        # I = np.vstack((It1x_k.ravel(), It1y_k.ravel())).T
        
        A = np.zeros((rows*cols,6))
        # print("Start inner loop")
        for i in range(rows):
            for j in range(cols):
                # Jacobian = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
                Jacobian = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
                A_sing = (I[cols*i+j].reshape((1,2))) @ Jacobian
                A[cols*i+j] = A_sing
        # print("End inner loop")
        H = (A.T) @ A
        sdp = (A.T) @ error_img
        dp = np.linalg.inv(H) @ sdp
        
        dp = np.reshape(dp,[1,6])
        
        #Update p
        p += dp
        
        #If delta_p is under the threshold break out early and return p
        if (np.linalg.norm(dp) < threshold):
            M = np.array([[1.0+p[0,0], p[0,2], p[0,4]], [p[0,1], 1.0+p[0,3], p[0,5]], [0, 0, 1]])
            return M
    

    M = np.array([[1.0+p[0,0], p[0,2], p[0,4]], [p[0,1], 1.0+p[0,3], p[0,5]], [0, 0, 1]])
    return M
