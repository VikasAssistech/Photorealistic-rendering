# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:44:51 2018

@author: Vikas upadhyay

This is implimentation of XDoG: Advanced Image Stylization with eXtended Difference-of-Gaussians
For DoG the preffered sigma=0.5 and k=1.6 , gamma =0.98
For xDoG the tuned sigma=0.5 and k=200 and gamma = 0.98 , epsilon =0.1 and phi=10
"""
import cv2
from skimage import img_as_uint
import numpy as np
from matplotlib import pyplot as plt



# extended version of DoG mentioned in paper reffering equation 5 
#  Dx(σ, k, τ ) = G(σ) − τ · G(k · σ)

def DoG(image,size=(0,0),k=1.6,sigma=0.5,gamma=0.98):
	image1 = cv2.GaussianBlur(image,size,sigma)          
	iamge2 = cv2.GaussianBlur(image,size,sigma*k)
	return (image1-gamma*iamge2)


# Threshold the dog image for edge detection as mentioned in equation 4
#    E(σ, k) = (1, if D0(σ, k) > 0; 0, otherwise)
def EdgeDog(img,sigma=0.5,k=200,gamma=0.98):
	Dx = DoG(img,sigma=sigma,k=k,gamma=0.98)
	for i in range(0,Dx.shape[0]):
		for j in range(0,Dx.shape[1]):
			if(Dx[i,j] > 0):
				Dx[i,j] = 255
			else:
				Dx[i,j] = 0
	return Dx

# Gary xdog including toning function
    
def XDoG_Garytone(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
	aux = DoG(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] >= epsilon):
				aux[i,j] = 1
			else:
				ht = np.tanh(phi*(aux[i][j] - epsilon))
				aux[i][j] = 1 + ht
	return aux*255


# thresholded DoG and XDoG
def DoG_XDoG_bool(grayImg):
    dog_img=EdgeDog(grayImg,sigma=1,k=1.6, gamma=0.98); 
    dog_bool_img=np.bool_(np.copy(np.uint8(dog_img)))
    xdog_img=XDoG_Garytone(grayImg,sigma=0.5,k=200, gamma=0.98); 
    ret, xdog_bool_img = cv2.threshold(np.uint8(xdog_img), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    xdog_bool_img=np.bool_(xdog_bool_img)
    print('DoG and xDoG thresholding done.................................')    
    return dog_bool_img, xdog_bool_img,dog_img, xdog_img




