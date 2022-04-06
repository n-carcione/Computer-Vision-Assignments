import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here    
    
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rows = It.shape[0]
    cols = It.shape[1]
    
    #Precompute:
    #Gradient of the template
    #use rectbivariatespline
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    It_spline = RectBivariateSpline(y, x, It)
    cc, rr = np.meshgrid(np.linspace(0,cols-1,cols), np.linspace(0,rows-1,rows))
    Itx = It_spline.ev(rr,cc, dy=1)
    Ity = It_spline.ev(rr,cc, dx=1)
    Its = It_spline.ev(rr,cc)
    I = np.vstack((Itx.ravel(), Ity.ravel())).T
    
    #Evaluate the Jacobian dW/dp at (x;0)
    A = np.zeros((rows*cols,6))
    for i in range(rows):
        for j in range(cols):
            Jacobian = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
            A_sing = (I[cols*i+j].reshape((1,2))) @ Jacobian
            #Compute the steepest descent images (A)
            A[cols*i+j] = A_sing

    #Compute the Hessian
    H = (A.T) @ A
    
    spline_It1 = RectBivariateSpline(y, x, It1)
    coords = np.vstack((cc.ravel(),rr.ravel(),np.ones((rows*cols))))
    mask = np.ones(It1.shape, dtype=int)
    mask_spline = RectBivariateSpline(y, x, mask)
    
    #Do the following for num_iters
    for iter in np.arange(num_iters):
        
        #Warp I with W(x; p) to compute I (W(x; p))
        coords_w = M @ coords
        It1_warped = spline_It1.ev(coords_w[1,:],coords_w[0,:])
        It1_warped = It1_warped.reshape(It.shape)
        mask_warped = mask_spline.ev(coords_w[1,:],coords_w[0,:])
        mask_warped = mask_warped.reshape(It.shape)
        
        #Compute the error image
        error_img = It1_warped - Its*mask_warped
        error_img = error_img.reshape(-1,1)
        
        #Compute steepest descent parameters
        sdp = (A.T) @ error_img
        #Compute delta_p
        dp = np.linalg.inv(H) @ sdp
        dp = dp.reshape(1,6)
        
        #Update the warp
        dM = np.array(([[1.0+dp[0,0], dp[0,1], dp[0,2]], [dp[0,3], 1.0+dp[0,4], dp[0,5]], [0.0, 0.0, 1.0]]))
        M = M @ np.linalg.inv(dM)
        
        # print(np.linalg.norm(dp))
        if (np.linalg.norm(dp) < threshold):
            return M
    
    return M
