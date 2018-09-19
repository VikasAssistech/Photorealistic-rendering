# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:45:14 2018

@author: Vikas Upadhyay
"""
from argparse import ArgumentParser
from math import floor
from PIL import Image
import numpy as np
import cv2
from skimage import img_as_uint

#    def floyd_steinberg_dither(image_file):
#        """
#        https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
#        Pseudocode:
#        for each y from top to bottom
#           for each x from left to right
#              oldpixel  := pixel[x][y]
#              newpixel  := find_closest_palette_color(oldpixel)
#              pixel[x][y]  := newpixel
#              quant_error  := oldpixel - newpixel
#              pixel[x+1][y  ] := pixel[x+1][y  ] + quant_error * 7/16
#              pixel[x-1][y+1] := pixel[x-1][y+1] + quant_error * 3/16
#              pixel[x  ][y+1] := pixel[x  ][y+1] + quant_error * 5/16
#              pixel[x+1][y+1] := pixel[x+1][y+1] + quant_error * 1/16
#        find_closest_palette_color(oldpixel) = floor(oldpixel / 256)
#        """
#        image_file=cv2.imread("vikas.jpg",1)
def ErrorDiffusion(new_img,factor):
    new_img = new_img.convert('RGB')
    pixel = new_img.load()
    
    x_lim, y_lim = new_img.size
#    factor=8
    for y in range(1, y_lim):
        for x in range(1, x_lim):
            red_oldpixel, green_oldpixel, blue_oldpixel = pixel[x, y]
        
            red_newpixel = floor(255/factor)*floor(factor*red_oldpixel/255)
            green_newpixel = floor(255/factor)*floor(factor*green_oldpixel/255)
            blue_newpixel = floor(255/factor)*floor(factor*blue_oldpixel/255)
    
            pixel[x, y] = red_newpixel, green_newpixel, blue_newpixel
    
            red_error = red_oldpixel - red_newpixel
            blue_error = blue_oldpixel - blue_newpixel
            green_error = green_oldpixel - green_newpixel
    
            if x < x_lim - 1:
                red = pixel[x+1, y][0] + round(red_error * 7/16)
                green = pixel[x+1, y][1] + round(green_error * 7/16)
                blue = pixel[x+1, y][2] + round(blue_error * 7/16)
                
                pixel[x+1, y] = (red, green, blue)
    
            if x > 1 and y < y_lim - 1:
                red = pixel[x-1, y+1][0] + round(red_error * 3/16)
                green = pixel[x-1, y+1][1] + round(green_error * 3/16)
                blue = pixel[x-1, y+1][2] + round(blue_error * 3/16)
                
                pixel[x-1, y+1] = (red, green, blue)
    
            if y < y_lim - 1:
                red = pixel[x, y+1][0] + round(red_error * 5/16)
                green = pixel[x, y+1][1] + round(green_error * 5/16)
                blue = pixel[x, y+1][2] + round(blue_error * 5/16)
                
                pixel[x, y+1] = (red, green, blue)
    
            if x < x_lim - 1 and y < y_lim - 1:
                red = pixel[x+1, y+1][0] + round(red_error * 1/16)
                green = pixel[x+1, y+1][1] + round(green_error * 1/16)
                blue = pixel[x+1, y+1][2] + round(blue_error * 1/16)
                
                pixel[x+1, y+1] = (red, green, blue)
                
#    out=np.uint8(new_img)
#    new_img.save('output_fs.jpg')
#    new_img.show()
    return new_img
  
#    
#new_img = Image.open('Lenna.png')
#out= FnS(new_img)      
#cv2.imwrite('output3.jpg',out)
#cv2.imshow('output', out)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#      