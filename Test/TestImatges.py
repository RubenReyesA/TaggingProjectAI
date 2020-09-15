# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:20:38 2019

@author: Pol
"""

import json
from skimage import io
from skimage.transform import rescale
from skimage import color
import numpy as np
import ColorNaming as cn
import matplotlib.pyplot as plt

import os.path

import Labels as lb
import KMeansMOD as km

"""
im=io.imread("Images/0007.png") #Inicialitzem im a la imatge
Opt={'verbose':False,'km:init':'first'}

plt.figure(1)
plt.imshow(im)
plt.axis('off')

k_m=km.KMeans(im,0,Opt)
k_m.run()
"""

im=io.imread("Images/0000.png")
k_m=km.KMeans(im,0)
