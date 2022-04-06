import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    mask = np.ones(image1.shape, dtype=bool)

    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    image2_w = affine_transform(image2, np.linalg.inv(M))
    
    # UNCOMMENT THE FOLLOWING IF YOU WANT TO USE INVERSE COMPOSITION
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    # image2_w = affine_transform(image2, np.linalg.inv(M))
    
    image2_w = binary_erosion(image2_w)
    image2_w = binary_dilation(image2_w)
    diff = np.abs(image1 - image2_w)
    mask = (diff > tolerance)
    
    return mask
