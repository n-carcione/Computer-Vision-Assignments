import numpy as np
#from PIL import Image

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
        
    #get the (row,col) size of each of the 3 images
    (r_row, r_col) = red.shape
    (g_row, g_col) = green.shape
    (b_row, b_col) = blue.shape
    
    #crop the three images to avoid the black borders interfering with the alignment
    #arbitrary amount to crop on each side of the image
    crop = 0.1
    r_crop = red[round(r_row*crop):round(r_row*(1-crop)),round(r_col*crop):round(r_col*(1-crop))]
    g_crop = green[round(g_row*crop):round(g_row*(1-crop)),round(g_col*crop):round(g_col*(1-crop))]
    b_crop = blue[round(b_row*crop):round(b_row*(1-crop)),round(b_col*crop):round(b_col*(1-crop))]
    
    #use the red image as the reference (shift green and blue to match red)
    #set up the defined ranges that the image can be shifted (-30 to 30 pixels)
    row_shifts = np.arange(-30,31)
    col_shifts = np.arange(-30,31)
    #define arbitrary, very large temp values for the ssd minimums
    min_ssd_g = 9999999999999
    min_ssd_b = 9999999999999
    
    #for each potential row shift amount (row and column)
    for row_shift in row_shifts:
        for col_shift in col_shifts:
            #apply the row and column shift to the cropped green and blue images
            g_roll = np.roll(g_crop,row_shift,axis=1)
            g_roll = np.roll(g_roll,col_shift,axis=0)
            b_roll = np.roll(b_crop,row_shift,axis=1)
            b_roll = np.roll(b_roll,col_shift,axis=0)
            
            #calculate the SSD between the cropped red image and the cropped, rolled green/blue image
            ssd_g = np.sum( np.sum( np.power( np.subtract(r_crop,g_roll),2 ) ) )
            ssd_b = np.sum( np.sum( np.power( np.subtract(r_crop,b_roll),2 ) ) )
            
            #if the SSD for either the green or blue image is less than the previous
            #SSD minimums, record the row and column shift that caused it and set
            #the current SSD to the new minimum for that color image
            if (ssd_g < min_ssd_g):
                green_row_shift = row_shift
                green_col_shift = col_shift
                min_ssd_g = ssd_g
                
            if (ssd_b < min_ssd_b):
                blue_row_shift = row_shift
                blue_col_shift = col_shift
                min_ssd_b = ssd_b
    
    #apply the shifts that caused the green and blue images to reach their
    #minimum SSDs to the entire green and blue images
    green = np.roll(green,green_row_shift,axis=1)
    green = np.roll(green,green_col_shift,axis=0)
    blue = np.roll(blue,blue_row_shift,axis=1)
    blue = np.roll(blue,blue_col_shift,axis=0)
    
    #stack the 3 r g b channels into one rgb image and return the rgb image
    rgb_output = np.dstack((red,green,blue))  # stacks 3 h x w arrays -> h x w x 3

    return rgb_output
