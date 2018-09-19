# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:31:40 2018

@author: vikas upadhyay
this part contain all possible color map
1: Histogram Popularity based color choice
2: K-mean based color choice
3: Histo
4: Median cut based color choice
3: 
"""
import cv2 
from PIL import Image
from skimage import img_as_uint
import numpy as np
from matplotlib import pyplot as plt
import sys
import colorgram

#  popularity based color selection
def FindPopularColormap(path,NoOfColor):
    img = Image.open(path)
    row,col=img.size
    colors = img.getcolors(row*col)    # return histogram of all possible colors exists in image 
    colors.sort();                     # sort the colors in order of popular colors
    colorMap=np.zeros((NoOfColor,3))   # create color map of wanted number of colors
    count=0
    for i in range (NoOfColor):
        colorMap[i,:]=colors[-i-1][1]
        count =count+colors[-i-1][0]
#        print(count)
    print('Histogram based popular colormap done.................................')
    return colorMap

# K-mean based color selection 
def FindColormapKmean(path,NoOfColor):
    img = Image.open(path)
    img = img.convert('P', dither=None,palette=Image.ADAPTIVE, colors=NoOfColor).convert("RGB")
    colors = img.getcolors()
    colors.sort(); cm=list(colors)    
    print('kmean colormap done.................................')
    return cm


# Get colormap using colorgram 
def FindColormapColorgram(path,NoOfColor):
    colors = colorgram.extract(path, NoOfColor)
    cm=np.zeros((NoOfColor,3))
    for i in range (NoOfColor):
        cm[i,:]=colors[i].rgb
    print('colorgram based colormap done.................................')
    return cm

#  median cut based color selection
def FindColormapMedianCut(rgb, gray, bit_count):
  """Runs the median cut algorithm."""
  lst=[]
  # Compute the cumulative sums.
  sums = np.cumsum(gray, axis=0, dtype=np.uint)
  sums = np.cumsum(sums, axis=1, dtype=np.uint)
 
  # Clone the rgb image.
  out = np.copy(rgb)

  def cut_rect(x0, y0, x1, y1, depth,lst):
      
    """Recursively splits a rectangle into areas of equal sum at the median of the longest axis. 
    This ensures that about the same number of colors is assigned to each of the new rectangle """
#    print(depth)
    if depth > bit_count:
       sliceRGB = rgb[x0:x1, y0:y1]
  # Draw a circle with the extracted diffuse colour.
       diff = np.median(sliceRGB, axis=(0, 1))
       colour = (int(diff[0]), int(diff[1]), int(diff[2])) 
       lst.append(colour)
       return lst
 
    total = sums[x1, y1] + sums[x0, y0] - sums[x0, y1] - sums[x1, y0]
    
    if x1 - x0 > y1 - y0:
      # Cut along y.
      best_x = x0
      best_d = sys.maxsize
      for x in range(x0, x1 + 1):
        sum = int(sums[ x, y1] + sums[x0, y0] - sums[x0, y1] - sums[ x, y0])
        if abs(2 * sum - total) < best_d:
          best_d = abs(2 * sum - total)
          print(best_d)
          best_x = x
      cv2.line(out, (y0, best_x), (y1, best_x), (0, 0, 255))
      cut_rect(x0, y0, best_x, y1, depth + 1,lst)
      cut_rect(best_x, y0, x1, y1, depth + 1,lst)
    else:
#      print('along x')
      # Cut along x.
      best_y = y0
      best_d = sys.maxsize
      for y in range(y0, y1 + 1):
        sum = int(sums[x1,  y] + sums[x0, y0] - sums[x0,  y] - sums[x1, y0])
        if abs(2 * sum - total) < best_d:
          best_d = abs(2 * sum - total)
          best_y = y
 
      cv2.line(out, (best_y, x0), (best_y, x1), (0, 0, 255))
      cut_rect(x0, y0, x1, best_y, depth + 1,lst)
      cut_rect(x0, best_y, x1, y1, depth + 1,lst)
 
  # Start the algorithm, covering the entire image.
  cut_rect(0, 0, gray.shape[0] - 1, gray.shape[1] - 1,0,lst)
  print('Median-cut based colormap done.................................')
  return lst