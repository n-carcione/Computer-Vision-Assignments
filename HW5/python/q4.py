import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bboxes_p = []
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    #Denoise image
    image_denoised = skimage.restoration.denoise_tv_chambolle(image,weight=0.1)
    #Convert to grayscale
    image_grey = skimage.color.rgb2gray(image_denoised)
    #Apply otsu threshold (makes pure black and white)
    thresh = skimage.filters.threshold_otsu(image_grey)
    im_thresh = skimage.morphology.closing(image_grey > thresh, skimage.morphology.square(3))
    #Label the characters as foreground
    im_label = skimage.measure.label(im_thresh,background=True)
    
    #Create bounding boxes around the characters
    for region in skimage.measure.regionprops(im_label):
    # take regions with large enough areas
        y1, x1, y2, x2 = region.bbox
        asp_rat = (x2-x1)/(y2-y1)
        area = (y2 - y1) * (x2 - x1)
        if area >= 1000 and area <= 10000000 and asp_rat < 1.75:
            # draw rectangle around segmented coins
            bboxes.append([y1,x1,y2,x2])
        #Save the bounding boxes that capture part of a character
        #These are defined as being smaller (but not too small) boxes
        #or boxes of the "full" size but they are longer than they are tall
        elif area < 1000 and area >= 100:
            bboxes_p.append([y1,x1,y2,x2])
        elif area >= 1000 and area <= 10000000 and asp_rat >= 1.75:
            bboxes_p.append([y1,x1,y2,x2])
    
    #Go through all of the partial bounding boxes to see if they should be part
    #of the bounding box around a more complete character
    for partial in bboxes_p:
        y1p, x1p, y2p, x2p = partial[0], partial[1], partial[2], partial[3]
        for full in bboxes:
            y1f, x1f, y2f, x2f = full[0], full[1], full[2], full[3]
            #The bottom of the partial aligns with the top of the full
            if np.abs(y2p-y1f) <= 30:
                #Left edge of partial left of full
                #Right edge of partial within full
                if (x1p <= x1f) and ((x2p <= x2f) and (x2p >= x1f)):
                    full[0] = y1p
                    full[1] = x1p
                    # print(1)
                #Left edge of partial left of full
                #Right edge of partial right of full (full within partial)
                elif (x1p <= x1f) and (x2p >= x2f):
                    full[0] = y1p
                    full[1] = x1p
                    full[3] = x2p
                    # print(2)
                #Left edge of partial within full
                #Right edge of partial right of full
                elif ((x1p <= x2f) and (x1p >= x1f)) and (x2p >= x2f):
                    full[0] = y1p
                    full[3] = x2p
                    # print(3)
                #Left edge of partial within full
                #Right edge of partial within full (partial within full)
                elif ((x1p <= x2f) and (x1p >= x1f)) and ((x2p <= x2f) and (x2p >= x1f)):
                    full[0] = y1p
                    # print(4)
            #The top of the partial aligns with the bottom of the full
            elif np.abs(y1p-y2f) <= 30:
                #Left edge of partial left of full
                #Right edge of partial within full
                if (x1p <= x1f) and ((x2p <= x2f) and (x2p >= x1f)):
                    full[2] = y2p
                    full[1] = x1p
                    # print(5)
                #Left edge of partial left of full
                #Right edge of partial right of full (full within partial)
                elif (x1p <= x1f) and (x2p >= x2f):
                    full[2] = y2p
                    full[1] = x1p
                    full[3] = x2p
                    # print(6)
                #Left edge of partial within full
                #Right edge of partial right of full
                elif ((x1p <= x2f) and (x1p >= x1f)) and (x2p >= x2f):
                    full[2] = y2p
                    full[3] = x2p
                    # print(7)
                #Left edge of partial within full
                #Right edge of partial within full (partial within full)
                elif ((x1p <= x2f) and (x1p >= x1f)) and ((x2p <= x2f) and (x2p >= x1f)):
                    full[2] = y2p
                    # print(8)
        
    
    bw = np.where(im_thresh == True, 1, 0)
    # bw = image_grey
    return bboxes, bw