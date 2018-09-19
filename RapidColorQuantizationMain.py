
"""
Created on Mon Aug 20 19:19:48 2018

@author: Vikas Upadhyay

"""

from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
import scipy.misc
from skimage import color
from matplotlib import pyplot as plt
# ColorPlalette return color table based on popularity and mediancut algorithum
from ColorPalette import FindPopularColormap, FindColormapColorgram, FindColormapKmean
from MedianCut import MedianCut
# Floyed and Steinberg error diffusion 
from FloyedSteinberg import ErrorDiffusion
# fusion of xDoG to quintized image to make it non- photorealistic rendered output
from StylizationFunctions import DoG,EdgeDog,XDoG_Garytone,DoG_XDoG_bool


# this is fast method to Compute minimum distances between one point and a set of points for quantization

def RapidQuantize(raster, MedianColor ):
    width, height, depth = raster.shape
    # reshape the image into 1 D array where each elemet contain R,G,B value
    reshaped_raster = np.reshape(raster, (width * height, depth))
    # label them based on minimum distance with colormap
    labels = pairwise_distances_argmin(reshaped_raster, MedianColor)
    # assigned the labled colors to image array and reshape to original image 
    quantized_raster = np.reshape(MedianColor[labels], (width, height, MedianColor.shape[1]))

    return quantized_raster


""" Stage 0: Read an image from a file as an array """
n_colors = 255
path='nine.jpg'
NoOfBit = np.log2(n_colors)
image = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (512, 512))
cv2.imwrite(path,image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
raster = scipy.misc.imread(path)
lab_raster = color.rgb2lab(raster)
rgb_raster = (color.lab2rgb(lab_raster) * 255).astype('uint8')
print(' Stage 0: Done.....')

"""  Step 1 : Running different algorithum for getting significant color palette : popularity, mediancut, kmean and colorgram """
Populararity = FindPopularColormap(path,n_colors)
Colorgram= FindColormapColorgram(path,n_colors)
Kmean=FindColormapKmean(path,n_colors)
# this will return median-cut centroid of box colors
medianCut, ColorCentroid = MedianCut(image,gray,NoOfBit-1)
MedianCut=np.zeros((n_colors,3))
for i in range(len(ColorCentroid)):
    MedianCut[i,:]=image[ColorCentroid[i][0],ColorCentroid[i][1],:]

print(' Stage 1: Done.....')

""" Step 2: Rapid quantization Quantization """
Quantized_medianCut=RapidQuantize(rgb_raster, MedianCut)
Quantized_medianCut = cv2.cvtColor(np.uint8(Quantized_medianCut), cv2.COLOR_BGR2RGB)

Quantized_Popularity=RapidQuantize(rgb_raster,Populararity)
Quantized_Popularity = cv2.cvtColor(np.uint8(Quantized_Popularity), cv2.COLOR_BGR2RGB)

cv2.imwrite('Quantized_medianCut.jpg',Quantized_medianCut);   Quantized_medianCutImg = Image.open("Quantized_medianCut.jpg")
cv2.imwrite('Quantized_Popularity.jpg',Quantized_Popularity);   Quantized_PopularityImg = Image.open("Quantized_Popularity.jpg")

print(' Stage 2: Done.....')


"""  Stage 3: Floyed and Steinberg Error difussion (Dithering) """
MedianCutFsImg = ErrorDiffusion(Quantized_medianCutImg,4);       MedianCutFsImg.save('MedianCutFsImg.jpg')
PopularFsImg = ErrorDiffusion(Quantized_PopularityImg,4);        PopularFsImg.save('PopularFsImg.jpg')

MedianCutFsImg=cv2.imread('MedianCutFsImg.jpg',1);      MedianCutFsDoGImg=np.copy(MedianCutFsImg)
PopularFsImg=cv2.imread('PopularFsImg.jpg',1);          PopularFsDoGImg=np.copy(PopularFsImg)
MedianCutFsxDoGImg=np.copy(MedianCutFsImg)             
PopularFsxDoGImg=np.copy(PopularFsImg)

print(' Stage 3: Done.....')


"""  Step 4: Apply xDoG Thresholding on outcome of Stage 3  """
dog_mask, xdog_mask,dog_img, xdog_img = DoG_XDoG_bool(gray)
for i in range(3):
    MedianCutFsDoGImg[:,:,i]=MedianCutFsDoGImg[:,:,i]*dog_mask
    MedianCutFsxDoGImg[:,:,i]=MedianCutFsxDoGImg[:,:,i]*xdog_mask
    PopularFsDoGImg[:,:,i]=PopularFsDoGImg[:,:,i]*dog_mask
    PopularFsxDoGImg[:,:,i]=PopularFsxDoGImg[:,:,i]*xdog_mask

cv2.imwrite('MedianCutFsDoGImg.jpg',MedianCutFsDoGImg)
cv2.imwrite('MedianCutFsxDoGImg.jpg',MedianCutFsxDoGImg)
cv2.imwrite('PopularFsDoGImg.jpg',PopularFsDoGImg)
cv2.imwrite('PopularFsxDoGImg.jpg',PopularFsxDoGImg)

print('Stage 4: Done ..........')


""" Results """
cv2.imshow('Original', image)
# after  quantization 
cv2.imshow('Quantized_medianCut', np.uint8(Quantized_medianCut))
cv2.imshow('Quantized_Popularity', Quantized_Popularity)
# after error diffusion
cv2.imshow('MedianCutFsImg', MedianCutFsImg)
cv2.imshow('PopularFsImg', PopularFsImg)
# after DoG 
cv2.imshow('MedianCutFsDoGImg', MedianCutFsDoGImg)
cv2.imshow('PopularFsDoGImg', PopularFsDoGImg)
# after xDoG
cv2.imshow('MedianCutFsxDoGImg', MedianCutFsxDoGImg)
cv2.imshow('PopularFsxDoGImg', PopularFsxDoGImg)
# xDoG output
cv2.imshow('xDoGImg', np.uint8(xdog_img))
cv2.imshow('DoGImg', np.uint8(dog_img))
cv2.waitKey(0)

cv2.destroyAllWindows()

#cv2.imwrite('QuantizedMedianCut.jpg',Quantized_medianCut)
