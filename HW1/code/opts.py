'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters for this homework
You are asked to play around with them for Q3.1
It is recommended that you leave them as they are before getting to Q3.1

You can change the values by changing their default fields or by command-line
arguments. For example, "python main.py --filter-scales 2 5 --K 50"
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW1: Scene Recognition')

    ## Paths
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='data folder')
    parser.add_argument('--feat-dir', type=str, default='../feat',
                        help='feature folder')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='output folder')

    # Visual words (requires tuning)
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4],
                        help='a list of scales for all the filters')
    
    parser.add_argument('--K', type=int, default=50,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=125,
                        help='Using only a subset of alpha pixels in each image') 

    ## Recognition system (requires tuning)
    parser.add_argument('--L', type=int, default=3,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')

    ## Additional options (add your own hyperparameters here)
    
    # this parameter was used for another extra credit attempt
    # parser.add_argument('--mid_mult', type=int, default=1,
    #                     help='Weight given to middle third of SPM layers, rest of image weighted at 1')
    
    parser.add_argument('--topx', type=int, default=10,
                        help='evaluation finds mode of the top "x" matches')

    ##
    opts = parser.parse_args()
    return opts
