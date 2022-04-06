# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:16:30 2021

@author: Nick Carcione
"""

from os.path import join

import numpy as np
from multiprocessing import Pool
from copy import copy
import statistics

import visual_words
import visual_recog
      

def find_mode_label(img_dists, train_labels, topx):
    '''
    Finds the most common scene type out of the top X training images that each
    test image was closest to

    [input]
    * img_dists     : the distance between each test image and each training image
                      a list of N histograms with length of T where N is the
                      # of test images and T is the # of training images
    * train_labels  : a list of the labels that corresponds to each training image
    * topx          : the "x" training images that each test was closest to

    [output]
    * pred_labels: numpy.ndarray of shape (N,) of the system's prediction
    '''
    # place the indices of the top x closest training images at the top of the array
    min_inds = np.argpartition(img_dists, topx-1, axis=1)
    # separate out these top x images from the rest
    min_inds = min_inds[:,0:topx]
    
    # for each test image, find the list of x labels that the test was closest
    # to and find the mode of that list. If there are multiple modes, return
    # the one that appears first (i.e. the one that has a training most alike
    # the test image). Assign the found mode as the system's prediction
    pred_labels = np.zeros(len(img_dists)).astype(int)
    for i in np.arange(len(img_dists)):
        min_labels = train_labels[min_inds[i,:]]
        pred_labels[i] = statistics.multimode(min_labels)[0]
    
    return pred_labels
        
    
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
    topx = opts.topx

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
    test_features = p.starmap(visual_recog.get_image_feature, args_feat)
    p.close()
    p.join()
    # Compute predicted label for each test image
    args_dist = list( (test_word_hist, features) for test_word_hist in test_features)
    p = Pool(n_worker)
    img_dists = p.starmap(visual_recog.distance_to_set, args_dist)
    p.close()
    p.join()
    
    # find the predictions by looking at the mode of the x closest training images
    pred_labels = find_mode_label(img_dists, train_labels, topx)

    # create the confusion matrix and calculate the accuracy
    conf = np.zeros((8,8))
    for i in range(len(test_labels)):
        conf[test_labels[i], pred_labels[i]] += 1
    accuracy = np.trace(conf) / np.sum(conf)
    
    return conf, accuracy
    

# the following are functions that were once used for extra credit but did not
# increase performance so they were scrapped. Kept here commented out for documentation/proof
# def use_avg_dist(img_dists, train_labels):
    train_0_inds = np.where(train_labels == 0)
    train_1_inds = np.where(train_labels == 1)
    train_2_inds = np.where(train_labels == 2)
    train_3_inds = np.where(train_labels == 3)
    train_4_inds = np.where(train_labels == 4)
    train_5_inds = np.where(train_labels == 5)
    train_6_inds = np.where(train_labels == 6)
    train_7_inds = np.where(train_labels == 7)
    
    pred_labels = np.zeros((len(img_dists))).astype(int)
    for i in np.arange(len(img_dists)):
        avg_0_dist = np.average(img_dists[i][train_0_inds])
        avg_1_dist = np.average(img_dists[i][train_1_inds])
        avg_2_dist = np.average(img_dists[i][train_2_inds])
        avg_3_dist = np.average(img_dists[i][train_3_inds])
        avg_4_dist = np.average(img_dists[i][train_4_inds])
        avg_5_dist = np.average(img_dists[i][train_5_inds])
        avg_6_dist = np.average(img_dists[i][train_6_inds])
        avg_7_dist = np.average(img_dists[i][train_7_inds])
        all_dists = np.array(([avg_0_dist, avg_1_dist, avg_2_dist, avg_3_dist, avg_4_dist, avg_5_dist, avg_6_dist, avg_7_dist]))
        pred_labels[i] = np.argmin(all_dists)
        
    return pred_labels


# def SPM_cust_weight_thirds(layer_hists, l, L, opts):
    '''
    Weight the given SPM layer such that the middle "third" is more important

    [input]
    * layer_hists   : array of histograms from SPM layer, shape (2^l, 2^l)
    * l             : layer number that layer_hists is from
    * L             : total number of layers in the SPM
    * opts          : options
    
    [output]
    layer_hists_w   : array of weighted SPM histograms, shape (4^l)
                      array has total histogram area of 1
    '''
    
    # this method does not apply to layers 0 and 1 since there is no "middle third"
    # just return back the SPM layer instead
    if (l==0 or l==1):
        return layer_hists.ravel() * 2**(-L) / 4**l
    
    # extract multiplier from opts
    mid_mult = opts.mid_mult
    # define an extra multiplier for the l=3 case that is between 1 & mid_mult
    l_3_extra_mult = (mid_mult + 1)/2
    
    # find the two middle indices of layer_hists
    mid_left = int( (2**l - 1)/2 )
    mid_right = mid_left + 1
    
    # calculate how wide the "middle third" should be
    mid_width = (2*round((2**l/3)/2))
    # calculate how many columns above/below middle indices are needed
    mid_extra = int((mid_width - 2) / 2)
    
    # initially set layer_hists_w equal to layer_hists
    layer_hists_w = layer_hists
    # apply the assigned multiplier to the "middle third"
    # note: a special procedure done when l=3 since the default way does not
    # capture enough of the middle
    if (l != 3):
        # mulitply block from mid_left - mid_extra to mid_right + mid_extra in each row
        # by the weighting for the middle third
        layer_hists_w[0:2**l, (mid_left-mid_extra):(mid_right+mid_extra+1)] *= mid_mult
        # calculate the total current area of histograms in the SPM layer
        # found by adding the extra weight given to the cells in the middle to the 
        # total number of cells
        divisor = 4**l + mid_width*(mid_mult-1)*2**l
        # unravel & adjust layer_hists_w so that it is normalized (whole thing = 1)
        layer_hists_w = layer_hists_w.ravel() / divisor
    else:
        # start off by following same logic/procedure as above
        layer_hists_w[0:2**l, (mid_left-mid_extra):(mid_right+mid_extra+1)] *= mid_mult
        # multiply columns 2 and 5 by the extra mult
        layer_hists_w[0:2**l, 2] *= l_3_extra_mult
        layer_hists_w[0:2**l, 5] *= l_3_extra_mult
        # calculate the unique divisor for the l = 3 case
        divisor = 4**l + 2*(mid_mult-1)*2**l + 2*(l_3_extra_mult - 1)*2**l
        # unravel & adjust layer_hists_w so that it is normalized (whole thing = 1)
        layer_hists_w = layer_hists_w.ravel() / divisor
    
    # apply the appropriate weighting to the layer according the SPM scheme
    layer_hists_w *= 2**(l-L-1)
    return layer_hists_w

# def get_feature_from_wordmap_SPM_cust(opts, wordmap):
#     '''
#     Compute histogram of visual words using spatial pyramid matching.

#     [input]
#     * opts      : options
#     * wordmap   : numpy.ndarray of shape (H,W)

#     [output]
#     * hist_all: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
#     '''
        
#     K = opts.K
#     L = opts.L
#     # ----- TODO -----
#     finest_layer_hists = np.zeros((4**L,K))
    
#     # break image up into cells for the finest layer
#     # get normalized histogram for each cell
#     #       store as an array with each cell's hist being a row
#     #       resulting in dimensions (# cells, K words)
#     rows = wordmap.shape[0]
#     cols = wordmap.shape[1]
#     num_rows_cell = int(rows / (2**L))
#     num_cols_cell = int(cols / (2**L))
#     for i in range(2**L):
#         for j in range(2**L):
#             finest_layer_hists[2**L*i+j,:] = visual_recog.get_feature_from_wordmap(opts, wordmap[i*num_rows_cell:(i+1)*num_rows_cell-1, j*num_cols_cell:(j+1)*num_cols_cell-1])
    
#     # start hist_all as the list of histograms from the finest layer
#     # these histograms are divided by the # of cells to normalize the layer's histogram
#     # the layer is then multiplied by the correct weight (always 1/2)
#     hist_all = SPM_cust_weight_thirds(finest_layer_hists, L, L, opts)
#     # combine cells to form cells of coarser layers by adding row vectors
#     #       together and dividing by 4
#     l = L - 1
#     prev_layer = finest_layer_hists
#     while (l >= 0):
#         ivals = (2 * np.arange( ( 2**(l+1) ) / 2 )).astype(int)
#         jvals = (np.arange( ( 2**(l+1) ) / 2 )).astype(int)
#         cell = 0
#         new_hist = np.zeros((4**l,K))
#         for i in ivals:
#             for j in jvals:
#                 cell_1 = prev_layer[(2**(l+1)*i + 2*j),:]
#                 cell_2 = prev_layer[(2**(l+1)*i + 2*j + 1),:]
#                 cell_3 = prev_layer[(2**(l+1)*(i+1) + 2*j),:]
#                 cell_4 = prev_layer[(2**(l+1)*(i+1) + 2*j + 1),:]
#                 new_hist[cell, :] = (cell_1 + cell_2 + cell_3 + cell_4) / 4
#                 cell = cell + 1
#         prev_layer = new_hist
#         # combine all hist vectors together into hist_all
#         hist_all = np.hstack((hist_all, SPM_cust_weight_thirds(new_hist, l, L, opts)))
#         l = l - 1
    
#     return hist_all
    
# def get_image_feature_cust(opts, img_path, dictionary):
#     '''
#     Extracts the spatial pyramid matching feature.

#     [input]
#     * opts      : options
#     * img_path  : path of image file to read
#     * dictionary: numpy.ndarray of shape (K, 3F)


#     [output]
#     * feature: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
#     '''

#     # ----- TODO -----
#     # load the image
#     data_dir = opts.data_dir
#     img_path = join(data_dir, img_path)
#     img = Image.open(img_path)
#     img = np.array(img)
    
#     # extract wordmap from the image
#     wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
#     # compute the SPM
#     feature = get_feature_from_wordmap_SPM_cust(opts, wordmap)
    
#     # return the computed feature
#     return feature

# def build_recognition_system_cust(opts, n_worker=1):
#     '''
#     Creates a trained recognition system by generating training features from all training images.

#     [input]
#     * opts        : options
#     * n_worker  : number of workers to process in parallel

#     [saved]
#     * features: numpy.ndarray of shape (N,M)
#     * labels: numpy.ndarray of shape (N)
#     * dictionary: numpy.ndarray of shape (K,3F)
#     * SPM_layer_num: number of spatial pyramid layers
#     '''

#     data_dir = opts.data_dir
#     out_dir = opts.out_dir
#     SPM_layer_num = opts.L

#     train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
#     train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
#     dictionary = np.load(join(out_dir, 'dictionary.npy'))

#     # ----- TODO -----
#     # get feature histograms for each image in the training set
#     #     use multiprocessing to do this to save time
#     #     do this by calling get_image_feature
#     # compile the feature histograms obtained for each image into one array
#     args = list( (opts, img_name, dictionary) for img_name in train_files)
#     p = Pool(n_worker)
#     features = p.starmap(get_image_feature_cust, args)
#     p.close()
#     p.join()

#     # example code snippet to save the learned system
#     np.savez_compressed(join(out_dir, 'trained_system_cust.npz'),
#         features=features,
#         labels=train_labels,
#         dictionary=dictionary,
#         SPM_layer_num=SPM_layer_num,
#     )

# def evaluate_recognition_system_cust(opts, n_worker=1, cust=False):
#     '''
#     Evaluates the recognition system for all test images and returns the confusion matrix.

#     [input]
#     * opts        : options
#     * n_worker  : number of workers to process in parallel

#     [output]
#     * conf: numpy.ndarray of shape (8,8)
#     * accuracy: accuracy of the evaluated system
#     '''

#     data_dir = opts.data_dir
#     out_dir = opts.out_dir

#     trained_system = np.load(join(out_dir, 'trained_system_cust.npz'))
#     dictionary = trained_system['dictionary']
#     features = trained_system['features']
#     train_labels = trained_system['labels']

#     # using the stored options in the trained system instead of opts.py
#     test_opts = copy(opts)
#     test_opts.K = dictionary.shape[0]
#     test_opts.L = trained_system['SPM_layer_num']

#     test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
#     test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

#     # ----- TODO -----
#     # computes the word histograms/features for each test image
#     args_feat = list( (opts, img_name, dictionary) for img_name in test_files)
#     p = Pool(n_worker)
#     test_features = p.starmap(get_image_feature_cust, args_feat)
#     p.close()
#     p.join()
#     # Compute predicted label for each test image
#     args_dist = list( (test_word_hist, features) for test_word_hist in test_features)
#     p = Pool(n_worker)
#     img_dists = p.starmap(visual_recog.distance_to_set, args_dist)
#     p.close()
#     p.join()
    
    
#     min_dists_inds = np.argmin(img_dists, axis=1)
#     pred_labels = train_labels[min_dists_inds]
    
#     conf = np.zeros((8,8))
#     for i in range(len(test_labels)):
#         conf[test_labels[i], pred_labels[i]] += 1
#     accuracy = np.trace(conf) / np.sum(conf)
    
#     return conf, accuracy