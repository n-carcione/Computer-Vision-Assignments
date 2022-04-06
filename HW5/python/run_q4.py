import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

i = 0
for img in os.listdir('../images'):
    #Plot black and white version of image overlayed with bounding boxes
    #around the characters
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    plt.figure()
    plt.imshow(bw,cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    # FIND THE ROWS
    #Find minimum y value in bboxes
    #Go across from left side (0) to right side at identified y
    #Search for any top left corners within +/- 100 pixels of the row
    #Each corner identified, add that bbox to a list of bboxes
    #Remove that bbox from the total list of bboxes
    #Add list of bboxes in the rows to a list of lists
    #Repeat until bboxes is empty
    if i == 0:
        width = 125
    else:
        width = 200
    chars_row = []
    total_chars = []
    bboxes_save = bboxes.copy()
    while bool(bboxes):
        y_lst = [itemy[0] for itemy in bboxes]
        y_val = y_lst[np.argmin(y_lst)]
        i = 0
        while i < len(bboxes):
            if np.abs(bboxes[i][0] - y_val) <= width:
                chars_row.append(bboxes[i])
                popped = bboxes.index(bboxes[i])
                bboxes.pop(popped)
            else:
                i += 1
        chars_row = sorted(chars_row, key=lambda x: x[1])
        total_chars.append(chars_row)
        chars_row = []    
    
    
    # load the weights
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    #Crop each image in each row one by one
    for row in total_chars:
        st = ''
        row_chars = np.array([])
        i = 0
        #Get the bw image of the character
        #Pad with white space (1s) around the edge to center the character in the image
        #Resize the character image to the 32x32 used to train the network
        #Transpose image to make it match training data format
        #Erode the image to make the character lines thicker and better connected
        #Flatten the image into a row vector
        #Divide by the maximum to get a better 0-->1 range
        for char in row:
            y1, x1, y2, x2 = char[0], char[1], char[2], char[3]
            text = bw[y1:y2,x1:x2]
            text = np.pad(text, (20,20), 'constant', constant_values=(1,1))
            text = skimage.transform.resize(text, (32,32), anti_aliasing=False)
            text = text.T
            text = skimage.morphology.erosion(text)
            text = text.flatten()
            text /= max(text)
            #Compile all of the characters in a row into one matrix
            if i == 0:
                row_chars = text
                i = 1
            else:
                row_chars = np.vstack((row_chars, text))
                
        # run the crops of characters in row through your neural network and
        #print them out
        h1 = forward(row_chars,params,'layer1',activation=sigmoid)
        probs = forward(h1,params,'output',activation=softmax)
        guesses = np.argmax(probs,1)
        for letter in guesses:
            st += letters[letter]
        print(st)
    print("----------------------------")
    