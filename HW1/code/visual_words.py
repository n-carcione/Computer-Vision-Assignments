import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from opts import get_opts
from multiprocessing import Pool
import sklearn.cluster


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    height = img.shape[0]
    width = img.shape[1]
    num_filter_types = 4
    F = num_filter_types * len(opts.filter_scales)
    
    # Check to make sure the image is float type and that its values are within [0, 1]
    if (img.dtype != 'float32'):
        img = img.astype(np.float32)
    if (img.min() < 0 or img.max() > 1):
        img = img/255
    
    # If the image is grayscale, copy the single grayscale channel to 3 channels
    if (img.ndim == 2):
        img = np.pad(img[...,None], ((0,0),(0,0),(0,2)))
        img[:,:,1] = img[:,:,0]
        img[:,:,2] = img[:,:,0]
    
    # Convert the rgb image to lab color space
    img = skimage.color.rgb2lab(img)
    
    filter_responses = np.zeros((height, width, 3*F))

    for i in range(len(opts.filter_scales)):
        # Gaussian filter applied to all 3 channels
        filter_responses[...,12*i] = scipy.ndimage.gaussian_filter(img[...,0], opts.filter_scales[i])
        filter_responses[...,12*i+1] = scipy.ndimage.gaussian_filter(img[...,1], opts.filter_scales[i])
        filter_responses[...,12*i+2] = scipy.ndimage.gaussian_filter(img[...,2], opts.filter_scales[i])
        # Laplace of the Gaussian filter applied to all 3 channels
        filter_responses[...,12*i+3] = scipy.ndimage.gaussian_laplace(img[...,0], opts.filter_scales[i])
        filter_responses[...,12*i+4] = scipy.ndimage.gaussian_laplace(img[...,1], opts.filter_scales[i])
        filter_responses[...,12*i+5] = scipy.ndimage.gaussian_laplace(img[...,2], opts.filter_scales[i])
        # Derivative of the Gaussian filter in the x direction applied to all 3 channels
        filter_responses[...,12*i+6] = scipy.ndimage.gaussian_filter(img[...,0], opts.filter_scales[i], (0,1))
        filter_responses[...,12*i+7] = scipy.ndimage.gaussian_filter(img[...,1], opts.filter_scales[i], (0,1))
        filter_responses[...,12*i+8] = scipy.ndimage.gaussian_filter(img[...,2], opts.filter_scales[i], (0,1))
        # Derivative of the Gaussian filter in the y direction applied to all 3 channels
        filter_responses[...,12*i+9] = scipy.ndimage.gaussian_filter(img[...,0], opts.filter_scales[i], (1,0))
        filter_responses[...,12*i+10] = scipy.ndimage.gaussian_filter(img[...,1], opts.filter_scales[i], (1,0))
        filter_responses[...,12*i+11] = scipy.ndimage.gaussian_filter(img[...,2], opts.filter_scales[i], (1,0))
        
    return filter_responses

def gen_rand_pixel_locs(alpha, rows, cols):
    # generates an array of size alpha x 2 of random float values [0,1)
    rand_pixels = np.random.rand(alpha, 2)
    # scale the array values to the row and column size of the image
    rand_pixels[:,0] = (rand_pixels[:,0]*rows)
    rand_pixels[:,1] = (rand_pixels[:,1]*cols)
    # convert the floats into ints (always round down to avoid too large indices)
    # and convert the array into a list of 2 element tuples (coordinates)
    rand_pixels = list(map(tuple,rand_pixels.astype(int)))
    
    return rand_pixels

def compute_dictionary_one_image(img_name):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    opts = get_opts()
    
    # open image and get size data for the image
    img_path = join(opts.data_dir, img_name)
    img = Image.open(img_path)
    img = np.array(img)
    rows = img.shape[0]
    cols = img.shape[1]
    
    # get filter responses for the image by calling function above
    filter_responses = extract_filter_responses(opts, img)
    
    # generate alpha random pixel locations in the image
    # rand_pixels structured in the format [row, column]
    rand_pixels = gen_rand_pixel_locs(opts.alpha, rows, cols)
    # ensures that alpha /unique/ pixels were generated i.e. no repeats
    while (opts.alpha != len(set(rand_pixels))):
        rand_pixels = gen_rand_pixel_locs(opts.alpha, rows, cols)
    
    # for each random pixel, sample each filtered response at that location
    #    results in alpha x 3F matrix
    rand_filter_resp = np.zeros((opts.alpha, filter_responses.shape[2]))
    for i in range(opts.alpha):
        for j in range(filter_responses.shape[2]):
            rand_filter_resp[i,j] = filter_responses[rand_pixels[i][0], rand_pixels[i][1], j]
    
    # save matrix to a temporary file
    out_file_name = img_name.replace('.jpg','_temp.npy')
    out_file_dir = '../dict_temp'
    np.save(join(out_file_dir, out_file_name), rand_filter_resp)

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    #use Pool with compute_dictionary_one_image and train_files list
    num_filters = 4
    F = num_filters * len(opts.filter_scales)
    p = Pool(n_worker)
    p.map(compute_dictionary_one_image, train_files)
    p.close()
    p.join()
    # combine all of the matrices from each image into alpha*T x 3F matrix
    filter_responses = np.zeros((1, 3*F))
    for name in train_files:
        name = join('../dict_temp', name.replace('.jpg','_temp.npy'))
        single = np.load(name)
        filter_responses = np.vstack((filter_responses, single))
    filter_responses = np.delete(filter_responses,0, 0)
    
    #cluster responses with k-means
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    # extract filter responses of img --> (H,W,3F)
    # compare each pixel in filtered response of img to the dictionary and
    # select the row that is closest to the pixel
    # return array with same (H,W) as img but each "pixel" value is the row/word
    # that the corresponding pixel in img is closest to
    
    height = img.shape[0]
    width = img.shape[1]
    num_words = dictionary.shape[0]
    filtered_images = extract_filter_responses(opts, img)
    depth = filtered_images.shape[2]
    filtered_unraveled = np.ravel(filtered_images)
    filtered_unraveled = filtered_unraveled.reshape((height*width,depth))
    
    distances = np.zeros((height*width, num_words))
    for i in range(num_words):
        distances[:,i] = np.transpose(scipy.spatial.distance.cdist(filtered_unraveled, dictionary[i].reshape(1,depth), 'euclidean'))
        
    wordmap = np.argmin(distances, axis=1).reshape(height,width)

    return wordmap

