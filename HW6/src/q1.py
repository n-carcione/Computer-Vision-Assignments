# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import skimage
import skimage.color
import cv2
import scipy.sparse
import scipy.sparse.linalg
from mpl_toolkits.mplot3d import Axes3D

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    #Initialize image frame to all zeros
    image = np.zeros((res))
    
    #Create width and height vectors centered around 0
    #Done like this sinze camera origin defined so that (x,y)=(0,0) is in middle
    #of frame, not the top left
    frame_width = np.arange(-res[1]/2,res[0]/2)
    frame_height = np.arange(-res[0]/2,res[1]/2)
    c_x = center[0]
    c_y = center[1]
    c_z = center[2]
    
    #For each pixel, decided if that pixel represents a point on the ball
    #If it does, compute the z-value and normal of that point
    #Then use the n-dot-l model to find the intensity at that pixel/spot
    #Intensity values are set to a minimum of 0
    for row in frame_height:
        for col in frame_width:
            x = col * pxSize
            y = row * pxSize
            if ((x-c_x)**2 + (y-c_y)**2) <= (rad**2):
                z = np.sqrt(rad**2 - (x-c_x)**2 - (y-c_y)**2) + c_z
                norm = np.array([2*x,2*y,2*z]) / np.sqrt(4*x**2+4*y**2+4*z**2)
                image[int(row+res[0]/2),int(col+res[1]/2)] = np.max((0,np.dot(norm,light)))
                
    
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    import cv2 as cv
    import skimage.color
    #Read in the first input image for array sizing
    #Read in as a uint16 rgb matrix
    in_im_path = path + "input_1.tif"
    image = cv.imread(in_im_path,-1)
    red = image[:,:,2]
    image[:,:,2] = image[:,:,0]
    image[:,:,0] = red
    I = np.zeros((7,image.shape[0]*image.shape[1]))
    s = (image.shape[0], image.shape[1])
    
    input_nums = np.array(([1, 2, 3, 4, 5, 6, 7]))
    for n in input_nums:
        #Read in each input image
        in_im_path = path + "input_" + str(n) + ".tif"
        image = cv.imread(in_im_path,-1)
        #Convert from cv2 BGR format to RGB
        red = image[:,:,2]
        image[:,:,2] = image[:,:,0]
        image[:,:,0] = red
        #Convert from RGB to XYZ
        image_xyz = skimage.color.rgb2xyz(image)
        #Luminance channel is the Y channel in XYZ (index 1)
        lum_chan = image_xyz[:,:,1]
        lum_vec = lum_chan.ravel()
        I[n-1,:] = lum_vec
    
    #Read in the light source directions
    source_file = path + "sources.npy"
    L = np.load(source_file)
    L = L.T

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    
    #Use the Moore-Penrose pseudo inverse equation to find B as shown in class
    S = L.T
    mp_pseudo = np.linalg.inv(S.T @ S) @ S.T
    B = mp_pseudo @ I

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    
    #Place holders for the albedos and normals
    albedos = np.zeros(B.shape[1])
    normals = np.zeros(B.shape)
    
    #For each column in the pseudoinverse, find the length of the 3D vector
    #represented by that column (this is the albedo)
    #Then divide that column (vector) by the length to get the unit normal
    for col in np.arange(B.shape[1]):
        albedos[col] = np.linalg.norm(B[:,col])
        normals[:,col] = B[:,col] / albedos[col]
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    
    #Reshape the albedo image to the correct frame/image size
    albedoIm = albedos.reshape(s)
    #Placeholder normal image values
    normalIm = np.zeros((s[0],s[1],3))
    #Go through each unit normal and, for that pixel, set the x-comp to red,
    #y-comp to green, and z-comp to blue.  This assumes normals is arranged
    #so that pixels are ordered across rows then down columns
    for row in np.arange(s[0]):
        for col in np.arange(s[1]):
            normalIm[row,col,0] = np.abs(normals[0,row*s[1]+col])
            normalIm[row,col,1] = np.abs(normals[1,row*s[1]+col])
            normalIm[row,col,2] = np.abs(normals[2,row*s[1]+col])
    
    #Normalize all values between 0 and 1
    albedoIm /= np.max(albedoIm)
    normalIm /= np.max(normalIm)
    
    #Show images
    plt.figure()
    plt.imshow(albedoIm, cmap='gray')
    
    plt.figure()
    plt.imshow(normalIm, cmap='rainbow')

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    #Placeholder values
    zx = np.zeros(s)
    zy = np.zeros(s)
    
    #For each column of normals (each pixel), calculate the partial derivative
    #of the surface with respect to x and y using the normal components
    for row in np.arange(s[0]):
        for col in np.arange(s[1]):
            zx[row,col] = -normals[0,row*s[1]+col] / normals[2,row*s[1]+col]
            zy[row,col] = -normals[1,row*s[1]+col] / normals[2,row*s[1]+col]
    
    #Integrate the partial derivatives using Frankot-Chellapa algorithm
    surface = integrateFrankot(zx, zy)

    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    
    #Create X and Y matrices that correlate to the pixel locations
    x = np.arange(surface.shape[1])
    y = np.arange(surface.shape[0])
    X,Y = np.meshgrid(x,y)
    #invert surface values to get a "positive" plot
    surface = -surface
    
    #Plot the 3D surface
    fig = plt.figure()
    ax3 = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax3)
    ax3.plot_surface(X, Y, surface, cmap="coolwarm")

    pass


if __name__ == '__main__':

    # Put your main code here
    #Code for rendering the sphere images
    #Takes a while to run, so only uncomment when necessary
    # center = np.array([0,0,0])
    # radius = 0.75/100
    # light = [np.array([1,1,1])/np.sqrt(3), np.array([1,-1,1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)]
    # pixel_size = 7 / 1000000
    # res = np.array([2160, 3840])
    # for direct in light:
    #     im = renderNDotLSphere(center, radius, direct, pixel_size, res)
    #     plt.figure()
    #     plt.imshow(im, cmap='gray')
    #     t = "Render for Source Direction " + np.array2string(direct)
    #     plt.title(t)
    
    #Code for running above functions to generate albedo and normal images and
    #surface plot
    I, L, s = loadData()
    
    #1 (d)
    U, S, Vh = np.linalg.svd(I,full_matrices=False)
    
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)
