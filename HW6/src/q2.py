# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    #Compute the SVD of I
    U, S, Vh = np.linalg.svd(I,full_matrices=False)
    S = np.diag(S)
    
    #Keep only the first 3 columns of U, upper left (3x3) of S, and first 3 rows of Vh
    U_hat = U[:,0:3]
    S_hat = S[0:3,0:3]
    Vh_hat = Vh[0:3,:]
    
    #Compute B and L using the above matrices.  The split of S is arbitrary.
    B = np.sqrt(S_hat) @ Vh_hat
    L = U_hat @ np.sqrt(S_hat)
    L = L.T

    return B, L


if __name__ == "__main__":

    I, L0, s = loadData()
    Bh, Lh = estimatePseudonormalsUncalibrated(I)
    # 2 (b-d)
    albedos, normals = estimateAlbedosNormals(Bh)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)
    
    # 2 (e)
    B_int = enforceIntegrability(Bh, s)
    albedos_int, normals_int = estimateAlbedosNormals(B_int)
    albedoIm_int, normalIm_int = displayAlbedosNormals(albedos_int, normals_int, s)
    surface_int = estimateShape(normals_int, s)
    plotSurface(surface_int)
    
    # 2 (f)
    mu = 0
    nu = 0
    lam = 1
    G = np.array(([[1, 0, 0], [0, 1, 0], [mu, nu, lam]]))
    B_int = np.linalg.inv(G).T @ B_int
    albedos_int, normals_int = estimateAlbedosNormals(B_int)
    # albedoIm_int, normalIm_int = displayAlbedosNormals(albedos_int, normals_int, s)
    surface_int = estimateShape(normals_int, s)
    plotSurface(surface_int)