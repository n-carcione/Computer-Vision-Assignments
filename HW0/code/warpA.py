import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    
    #num_rows = output_shape[0]
    #num_cols = output_shape[1]
    
    #**************************************************************************
    #Following commented out code block reaches same result using for loops
    #**************************************************************************
    
    # A_inv = np.linalg.inv(A)
    # warp_im = np.zeros((output_shape[0], output_shape[1]))
    
    # for row in range(output_shape[0]):
    #     for col in range(output_shape[1]):
    #         source_pt = np.dot(A_inv,np.array([[row, col, 1]]).T)
    #         source_row = round(source_pt[0,0])
    #         source_col = round(source_pt[1,0])
    #         if 0 <= source_col < output_shape[1] and 0 <= source_row < output_shape[0]:
    #             warp_im[row][col] = im[source_row][source_col]
    #**************************************************************************
    
    #Calculate the inverse of the affine matrix A. Multiplying A_inv by the
    #coordinates of the warped image undoes the warping and finds the corresponding
    #pixel from the source image
    A_inv = np.linalg.inv(A)
    
    #Create a meshgrid that results in 2 matrices where B represents row #
    #and C represents the column #.  Both are size 200x150
    C, B = np.meshgrid(np.arange(0,150),np.arange(0,200))
    
    #Calculate 200x150 matrices that represent the row # and column # of the pixel
    #in the source image that is the result of unwarping each destination pixel
    rs = A_inv[0,0]*B + A_inv[0,1]*C + A_inv[0,2]
    cs = A_inv[1,0]*B + A_inv[1,1]*C + A_inv[1,2]
    
    #The values in rs and cs represent row and column #'s, therefore they should be
    #integers. These are kept in array form to be used later for image cleaning
    rs = np.around(rs).astype(int)
    cs = np.around(cs).astype(int)
    
    #The source row and column arrays are unraveled into one long vector each
    #These vectors are cleaned up to set any values <0 or >= the number of rows/
    #columns to 0, since these result in invalid row/column #'s
    #While setting these to 0 results in garbage gray scale values being calculated
    #at these r-c pairs, the location of these is preserved by rs and cs, and they
    #are later fixed
    #A (deep) copy of rs and cs must be made for rsl and csl so that fixing
    #rsl and csl does not affect rs and cs
    rsl = rs.copy().ravel()
    rsl[rsl < 0] = 0
    rsl[rsl >= output_shape[0]] = 0
    
    csl = cs.copy().ravel()
    csl[csl < 0] = 0
    csl[csl >= output_shape[1]] = 0
    
    #The warp image is created by taking the gray scale value of the source image
    #at all of the row-column pairs found.  The original rs and cs arrays must be
    #in vector form to do this, hence rsl and csl
    #The warped image is then reshaped from a vector to an array of the desired size
    warp_im = im[rsl, csl]
    warp_im = warp_im.reshape(output_shape[0], output_shape[1])
    
    #The final warped image is then cleaned up. Anywhere there is a row/column
    #value <0 or >= the total number of rows/columns in the original rs and cs
    #arrays, the value in the final warped image is set to 0 (black), eliminating
    #any junk values from setting invalid values to 0 in rsl and csl
    warp_im[rs < 0] = 0
    warp_im[rs >= output_shape[0]] = 0
    warp_im[cs < 0] = 0
    warp_im[cs >= output_shape[1]] = 0
    
    return warp_im