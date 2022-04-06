from alignChannels import alignChannels
import numpy as np
from PIL import Image

# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load('C:/Users/Nick Carcione/Desktop/CMU/Computer Vision/HW0/data/red.npy')
green = np.load('C:/Users/Nick Carcione/Desktop/CMU/Computer Vision/HW0/data/green.npy')
blue = np.load('C:/Users/Nick Carcione/Desktop/CMU/Computer Vision/HW0/data/blue.npy')

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
im = Image.fromarray(rgbResult)
im.save('C:/Users/Nick Carcione/Desktop/CMU/Computer Vision/HW0/results/rgb_output.jpg', 'JPEG')