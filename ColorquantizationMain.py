# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:28:09 2018

@author: vikas upadhyay, Assistech Lab, SIT

"""

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# ColorPlalette return color table based on popularity and mediancut algorithum
from ColorPalette import FindPopularColormap, FindColormapColorgram, FindColormapKmean
from MedianCut import MedianCut
# Floyed and Steinberg error diffusion 
from FloyedSteinberg import ErrorDiffusion
# fusion of xDoG to quintized image to make it non- photorealistic rendered output
from StylizationFunctions import DoG,EdgeDog,XDoG_Garytone,DoG_XDoG_bool




# Quantization based on nearest neighobure exhaustive search algorithum hence will take time
def colorquantization(img,popularity,mediancut,colorgram,kmean):
    img1=np.copy(img); img2=np.copy(img);img3=np.copy(img);img4=np.copy(img)
    bp=img1[:,:,0];  gp=img1[:,:,1];  rp=img1[:,:,2]
    bmc=img2[:,:,0]; gmc=img2[:,:,1]; rmc=img2[:,:,2]
    bcg=img3[:,:,0]; gcg=img3[:,:,1]; rcg=img3[:,:,2]
    bkm=img4[:,:,0]; gkm=img4[:,:,1]; rkm=img4[:,:,2]
    row,col,l = img.shape
    
    PopularityImg=np.zeros((row,col,l)); EuclDistPopularity=np.zeros((1,len(popularity)))
    MedianCutImage=np.zeros((row,col,l)); MCutDistPopularity=np.zeros((1,len(mediancut)))
    ColorGramImage=np.zeros((row,col,l)); ColorGramDistPopularity=np.zeros((1,len(colorgram)))
    KmeanImage=np.zeros((row,col,l)); KmeanDistPopularity=np.zeros((1,len(kmean)))
    
    for i in range(0,row):
        print(np.round(100*i/row))
        for j in range(0,col):
            for k in range(0,len(popularity)):
                KmeanColor = kmean[-1-k][1]           # for kmean colormap
                PopularColor = popularity[-1-k]    # for popularity colormap
                MCutColor = mediancut[-1-k]       # for mediancut colormap
                CgramColor = colorgram[-1-k]       # for colorgram colormap
                EuclDistPopularity[:,k] = (PopularColor[0] - rp[i,j])**2+(PopularColor[1] - gp[i,j])**2+(PopularColor[2] - bp[i,j])**2    
                MCutDistPopularity[:,k] = (MCutColor[0] - rmc[i,j])**2+(MCutColor[1] - gmc[i,j])**2+(MCutColor[2] - bmc[i,j])**2  
                ColorGramDistPopularity[:,k] = (CgramColor[0] - rcg[i,j])**2+(CgramColor[1] - gcg[i,j])**2+(CgramColor[2] - bcg[i,j])**2  
                KmeanDistPopularity[:,k] = (KmeanColor[0] - rkm[i,j])**2+(KmeanColor[1] - gkm[i,j])**2+(KmeanColor[2] - bkm[i,j])**2  
            PopularDistMin =np.argmin(EuclDistPopularity);    PopularColor = popularity[-1-PopularDistMin]
            MCutDistMin=np.argmin(MCutDistPopularity);        MCutColor = mediancut[-1-MCutDistMin]      
            CgDistMin=np.argmin(ColorGramDistPopularity);     CgramColor = colorgram[-1-CgDistMin] 
            KmDistMin=np.argmin(KmeanDistPopularity);         KmeanColor = kmean[-1-KmDistMin][1]                  
            rp[i,j]= PopularColor[0];   gp[i,j]= PopularColor[1];   bp[i,j]= PopularColor[2]
            rmc[i,j]= MCutColor[0];     gmc[i,j]= MCutColor[1];     bmc[i,j]= MCutColor[2]
            rcg[i,j]= CgramColor[0];    gcg[i,j]= CgramColor[1];    bcg[i,j]= CgramColor[2]
            rkm[i,j]= KmeanColor[0];    gkm[i,j]= KmeanColor[1];    bkm[i,j]= KmeanColor[2]
    PopularityImg[:,:,0]=bp;        PopularityImg[:,:,1]=gp;        PopularityImg[:,:,2]=rp
    MedianCutImage[:,:,0]=bmc;      MedianCutImage[:,:,1]=gmc;      MedianCutImage[:,:,2]=rmc
    ColorGramImage[:,:,0]=bcg;      ColorGramImage[:,:,1]=gcg;      ColorGramImage[:,:,2]=rcg
    KmeanImage[:,:,0]=bkm;          KmeanImage[:,:,1]=gkm;          KmeanImage[:,:,2]=rkm
    print('color quantization done')
    return PopularityImg, MedianCutImage, ColorGramImage, KmeanImage


""" Step 0: getting image and possible number of colors for quantization """
NoOfColor=255
path='nine.jpg'
NoOfBit  = np.log2(NoOfColor)
RGB_img  = cv2.resize(cv2.imread(path,1), dsize=(512, 512), interpolation=cv2.INTER_CUBIC); image= np.copy(RGB_img)
cv2.imwrite(path,RGB_img)
Gray_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)


"""  Step 1 : Running different algorithum for getting significant color palette : popularity, mediancut, kmean and colorgram """
Populararity = FindPopularColormap(path,NoOfColor)
MedianColrs,ColorCentroid = MedianCut(RGB_img, Gray_img, NoOfBit-1)
MedianCutColor=np.zeros((NoOfColor,3))
for i in range(len(ColorCentroid)):
    MedianCutColor[i,:]=image[ColorCentroid[i][0],ColorCentroid[i][1],:]

Colorgram= FindColormapColorgram(path,NoOfColor)
Kmean=FindColormapKmean(path,NoOfColor)


""" Step 2: Quantization based on euclidean distance : This is exhaustive search algorithum hence will take time"""
PopularImg, MedianCutImg, ColorGramImg, KmeanImg = colorquantization(image,Populararity,MedianCutColor,Colorgram,Kmean)
cv2.imwrite('PopularImg.jpg',PopularImg);       PopularImage = Image.open("PopularImg.jpg")
cv2.imwrite('MedianCutImg.jpg',MedianCutImg);   MedianCutImage = Image.open("MedianCutImg.jpg")
cv2.imwrite('ColorGramImg.jpg',ColorGramImg);   ColorGramImage = Image.open("ColorGramImg.jpg")
cv2.imwrite('KmeanImg.jpg',KmeanImg);           KmeanImage = Image.open("KmeanImg.jpg")


""" Step 3: Floyed and Steinberg Error difussion (Dithering) """
PopularFsImg = ErrorDiffusion(PopularImage,4);          PopularFsImg.save('PopularFsImg.jpg')
MedianCutFsImg = ErrorDiffusion(MedianCutImage,4);      MedianCutFsImg.save('MedianCutFsImg.jpg')
ColorGramFsImg = ErrorDiffusion(ColorGramImage,4);      ColorGramFsImg.save('ColorGramFsImg.jpg')
KmeanFsImg = ErrorDiffusion(KmeanImage,4);              KmeanFsImg.save('KmeanFsImg.jpg')

# reading back the images after error diffusion
PopularFsDoGImg = cv2.imread('PopularFsImg.jpg');       PopularFsxDoGImg=np.copy(PopularFsDoGImg)
MedianCutFsDoGImg=cv2.imread('MedianCutFsImg.jpg');     MedianCutFsxDoGImg=np.copy(MedianCutFsDoGImg)
ColorGramFsDoGImg=cv2.imread('ColorGramFsImg.jpg');     ColorGramFsxDoGImg=np.copy(ColorGramFsDoGImg)
KmeanFsDoGImg=cv2.imread('KmeanFsImg.jpg');             KmeanFsxDoGImg=np.copy(KmeanFsDoGImg)


"""   Step 4: Apply xDoG Thresholding on outcome of Stage 3 """
dog_mask, xdog_mask,dog_img, xdog_img = DoG_XDoG_bool(Gray_img)
for i in range(3):
    PopularFsDoGImg[:,:,i]=PopularFsDoGImg[:,:,i]*dog_mask
    MedianCutFsDoGImg[:,:,i] = MedianCutFsDoGImg[:,:,i]*dog_mask
    ColorGramFsDoGImg[:,:,i] = ColorGramFsDoGImg[:,:,i]*dog_mask
    KmeanFsDoGImg[:,:,i] = KmeanFsDoGImg[:,:,i]*dog_mask    
    PopularFsxDoGImg[:,:,i] = PopularFsDoGImg[:,:,i]*xdog_mask
    MedianCutFsxDoGImg[:,:,i] = MedianCutFsDoGImg[:,:,i]*xdog_mask
    ColorGramFsxDoGImg[:,:,i] = ColorGramFsDoGImg[:,:,i]*xdog_mask
    KmeanFsxDoGImg[:,:,i] = KmeanFsDoGImg[:,:,i]*xdog_mask


cv2.imwrite('PopularFsDoGImg.jpg',PopularFsDoGImg);     cv2.imwrite('PopularFsxDoGImg.jpg',PopularFsxDoGImg);
cv2.imwrite('MedianCutFsDoGImg.jpg',MedianCutFsDoGImg); 
#cv2.imwrite('MedianCutFsxDoGImg.jpg',cv2.cvtColor(MedianCutFsxDoGImg, cv2.COLOR_BGR2RGB));
cv2.imwrite('ColorGramFsDoGImg.jpg',ColorGramFsDoGImg); cv2.imwrite('ColorGramFsxDoGImg.jpg',ColorGramFsxDoGImg);
cv2.imwrite('KmeanFsDoGImg.jpg',KmeanFsDoGImg);         cv2.imwrite('KmeanFsxDoGImg.jpg',KmeanFsxDoGImg);

""" results  """
cv2.imshow('original',RGB_img)
cv2.imshow('popularFs',np.uint8(cv2.imread('PopularFsImg.jpg')))
cv2.imshow('McutFs', np.uint8(cv2.imread('MedianCutFsImg.jpg')))
cv2.imshow('ColorgramFs', np.uint8(cv2.imread('ColorGramFsImg.jpg')))
cv2.imshow('KmeanFs', np.uint8(cv2.imread('KmeanFsImg.jpg')))


cv2.imshow('ColorgramFsxDog', np.uint8(ColorGramFsxDoGImg))
cv2.imshow('KmeanFsxDog', np.uint8(KmeanFsxDoGImg))
cv2.imshow('McutFsxDoG', np.uint8(MedianCutFsxDoGImg))
cv2.imshow('popularFsxDoG',np.uint8(PopularFsxDoGImg))

cv2.waitKey(0)
cv2.destroyAllWindows() 