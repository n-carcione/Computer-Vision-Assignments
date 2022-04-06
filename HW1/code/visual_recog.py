import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
from multiprocessing import Pool

import visual_words
# import custom


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    histogram = np.histogram( wordmap, bins=K, range=[0,K], density=True)
    hist = histogram[0]
    
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    finest_layer_hists = np.zeros((4**L,K))
    
    # break image up into cells for the finest layer
    # get normalized histogram for each cell
    #       store as an array with each cell's hist being a row
    #       resulting in dimensions (# cells, K words)
    rows = wordmap.shape[0]
    cols = wordmap.shape[1]
    num_rows_cell = int(rows / (2**L))
    num_cols_cell = int(cols / (2**L))
    for i in range(2**L):
        for j in range(2**L):
            finest_layer_hists[2**L*i+j,:] = get_feature_from_wordmap(opts, wordmap[i*num_rows_cell:(i+1)*num_rows_cell-1, j*num_cols_cell:(j+1)*num_cols_cell-1])
    
    # start hist_all as the list of histograms from the finest layer
    # these histograms are divided by the # of cells to normalize the layer's histogram
    # the layer is then multiplied by the correct weight (always 1/2)
    hist_all = finest_layer_hists.ravel() * 2**(-1) / 4**L
    # combine cells to form cells of coarser layers by adding row vectors
    #       together and dividing by 4
    l = L - 1
    prev_layer = finest_layer_hists
    while (l >= 0):
        ivals = (2 * np.arange( ( 2**(l+1) ) / 2 )).astype(int)
        jvals = (np.arange( ( 2**(l+1) ) / 2 )).astype(int)
        cell = 0
        new_hist = np.zeros((4**l,K))
        for i in ivals:
            for j in jvals:
                cell_1 = prev_layer[(2**(l+1)*i + 2*j),:]
                cell_2 = prev_layer[(2**(l+1)*i + 2*j + 1),:]
                cell_3 = prev_layer[(2**(l+1)*(i+1) + 2*j),:]
                cell_4 = prev_layer[(2**(l+1)*(i+1) + 2*j + 1),:]
                new_hist[cell, :] = (cell_1 + cell_2 + cell_3 + cell_4) / 4
                cell = cell + 1
        prev_layer = new_hist
        # combine each matrix of histograms for each layer into one long vector (.ravel)
        # divide each layer's hist vector by # cells and mult by weighting factor
        # combine all hist vectors together into hist_all
        if (l == 0 or l == 1):
            hist_norm = new_hist.ravel() / cell * 2**(-L)
            hist_all = np.hstack((hist_all, hist_norm))
        else:
            hist_norm = new_hist.ravel() / cell * 2**(l-L-1)
            hist_all = np.hstack((hist_all, hist_norm))
        l = l - 1
    
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    '''

    # ----- TODO -----
    # load the image
    data_dir = opts.data_dir
    img_path = join(data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img)
    
    # extract wordmap from the image
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
    # compute the SPM
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    # return the computed feature
    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    # get feature histograms for each image in the training set
    #     use multiprocessing to do this to save time
    #     do this by calling get_image_feature
    # compile the feature histograms obtained for each image into one array
    args = list( (opts, img_name, dictionary) for img_name in train_files)
    p = Pool(n_worker)
    features = p.starmap(get_image_feature, args)
    p.close()
    p.join()

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    * histograms: numpy.ndarray of shape (N,(K*(4^(L+1)-1)/3))

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    hist_dist = 1 - np.sum(np.minimum(word_hist, histograms),axis=1)
    return hist_dist    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    train_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    # computes the word histograms/features for each test image
    args_feat = list( (opts, img_name, dictionary) for img_name in test_files)
    p = Pool(n_worker)
    test_features = p.starmap(get_image_feature, args_feat)
    p.close()
    p.join()
    # Compute predicted label for each test image
    args_dist = list( (test_word_hist, features) for test_word_hist in test_features)
    p = Pool(n_worker)
    img_dists = p.starmap(distance_to_set, args_dist)
    p.close()
    p.join()
    
    # if (custom):
    #     pred_labels = custom.find_mode_label(img_dists, train_labels)
    # else:
    #     min_dists_inds = np.argmin(img_dists, axis=1)
    #     pred_labels = train_labels[min_dists_inds]
    min_dists_inds = np.argmin(img_dists, axis=1)
    pred_labels = train_labels[min_dists_inds]
    
    conf = np.zeros((8,8))
    for i in range(len(test_labels)):
        conf[test_labels[i], pred_labels[i]] += 1
    accuracy = np.trace(conf) / np.sum(conf)
    
    return conf, accuracy

