import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It_par, It1_par, rect_par, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    #Make copies of the inputs just to make sure nothing weird happens with
    #pass by reference
    p = p0.copy()
    It = It_par.copy()
    It1 = It1_par.copy()
    rect = rect_par.copy()
    
    #get size of the image
    rows_img = It1.shape[0]
    cols_img = It1.shape[1]
    
    #x and y values for rectangle corners
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    rows_rect = y2-y1
    cols_rect = x2-x1
    
    #arrays that detail each column and row of the images for the
    #RectBivariateSpline commands
    x = np.arange(0, cols_img)
    y = np.arange(0, rows_img)
    
    #Evaluate the spline of the template image (It) in the designated rectangle
    #If a full frame is given, have to do this process
    #If only the template is given in template correction case, just use that
    if (It.shape == It1.shape):
        temp_cols, temp_rows = np.meshgrid(np.linspace(x1, x2, cols_rect), np.linspace(y1, y2, rows_rect))
        spline = RectBivariateSpline(y, x, It)
        T = spline.ev(temp_rows, temp_cols)
    else:
        T = It
    
    #Create spline for the It1 image
    spline_It1 = RectBivariateSpline(y, x, It1)
    
    #Compute x- and y-direction gradients of original It1
    It1_x = np.gradient(It1, axis=1)
    It1_y = np.gradient(It1, axis=0)
    spline_x = RectBivariateSpline(y, x, It1_x)
    spline_y = RectBivariateSpline(y, x, It1_y)
    
    #"compute" the Jacobian dW/dp = [1 0; 0 1]
    Jacobian = np.array(([1, 0],[0, 1]))
    
    #Do the following for num_iters
    for iter in np.arange(num_iters):
        
        #x and y values of the warped rectangle
        x1w, x2w, y1w, y2w, = x1+p[0], x2+p[0], y1+p[1], y2+p[1]
        
        #Warp It1 by the warp W(x:p) to get It1_warped
        warp_cols, warp_rows = np.meshgrid(np.linspace(x1w,x2w,cols_rect), np.linspace(y1w,y2w,rows_rect))
        It1_warped = spline_It1.ev(warp_rows,warp_cols)
        
        #Subtract the warped image from the template to get an error image
        #Only use the specified rectangular region since that is all that is needed
        error_img = T - It1_warped
        error_img = error_img.reshape(-1,1)
        
        #Warp x- and y-direction gradiesnts by the warp W(x:p)
        It1x_warped = spline_x.ev(warp_rows, warp_cols)
        It1y_warped = spline_y.ev(warp_rows, warp_cols)
        #I is (m,2) where m is total # of pts in rectangle
        I = np.vstack((It1x_warped.ravel(), It1y_warped.ravel())).T
        
        #A is (m,2)
        A = I @ Jacobian
        #H is (2,m) @ (m,2) = (2,2)
        H = A.T @ A
        #dp is (2,2) @ (2,m) @ (m,1) = (2,1)
        dp = np.linalg.inv(H) @ (A.T) @ (error_img.reshape(-1,1))
        
        #Update p
        p[0] += dp[0]
        p[1] += dp[1]
        
        #If delta_p is under the threshold break out early and return p
        if (np.linalg.norm(dp) < threshold):
            return p
    
    return p
