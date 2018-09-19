# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:31:40 2018

@author: vikas upadhyay
"""

import cv2 
import numpy as np
import sys

"""Compute the pq moment just to find the centroide of rectangle"""  
def m(p, q, f):
  if p == 0 and q == 0:
    return np.sum(f)
 
  if p == 1:
    sum = 0
    for index, e in np.ndenumerate(f):   # index iterator
      x, y = index
      sum += x * e
    return sum
 
  if q == 1:
    sum = 0
    for index, e in np.ndenumerate(f):
      x, y = index
      sum += y * e
    return sum 
  return 0

 
def MedianCut(rgb, gray, bit_count):
  """Runs the median cut algorithm."""
  lst=[];centroid=[]
  # Compute the cumulative sums gives the greatest range among RGB
  sums = np.cumsum(gray, axis=0, dtype=np.uint)
  sums = np.cumsum(sums, axis=1, dtype=np.uint)
 
  # Clone the rgb image.
  out = np.copy(rgb)

  def cut_rect(x0, y0, x1, y1, depth,lst):
      
    """Recursively splits a rectangle into areas of equal sum."""
#    print(depth)
    if depth > bit_count:
       sliceRGB = rgb[x0:x1, y0:y1]
       sliceGray = gray[x0:x1, y0:y1] 
  # Find the centroid of rectengle.
       m00 = m(0, 0, sliceGray)
       cx = x0 + int(m(1, 0, sliceGray) / m00)
       cy = y0 + int(m(0, 1, sliceGray) / m00) 
  # Draw a circle with the extracted diffuse colour.
       diff = np.median(sliceRGB, axis=(0, 1))
       colour = (int(diff[0]), int(diff[1]), int(diff[2])) 
       lst.append(colour)
       centroid.append([cy,cx])
       return lst,centroid
 
    total = sums[x1, y1] + sums[x0, y0] - sums[x0, y1] - sums[x1, y0]
    print(total)
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
      best_d = sys.maxsize   # declear a maximum number
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
  return lst,centroid
 

#  
bit_count = 3
NoOfColor = 2**(bit_count+1)
image = cv2.resize(cv2.imread('one.jpg', cv2.IMREAD_COLOR), (512, 512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
image, cm = MedianCut(image,gray,bit_count)
# 
#cv2.imshow('output', image)
##cv.imwrite('output.png', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
 
