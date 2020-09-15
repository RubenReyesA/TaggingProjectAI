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

import os.path

import Labels as lb
import KMeans as km


#Dades inicials dels Punts i Centroides
X=np.array([[155,185,153],[282,13,286],[263,68,7],[76,209,53],[156,5,185],[260,234,50]])
C=np.array([[63,217,154],[265,136,97]])


#Exercici 1. Distancia euclidiana del punt 6 a tots els centroides
print("\nExercici 1.")
d=km.distance(X,C)
print(d)

#Exercici 2.
print("\nExercici 2.")
print(np.argmin(d,axis=1)+1) # Gràcies a la llibreria np.array, definim els índexs dels mínims per files.
        

#Exercici 3. 
print("\nExercici 3.")
k_m=km.KMeans(X,2)
k_m.centroids=C
k_m._iterate()
print(k_m.centroids)

#Exercici 6.
print("\nExercici 6.")
im=io.imread("Images/0017.png") #Inicialitzem im a la imatge
k_m=km.KMeans(im,3)
k_m.run()
print(k_m.centroids)


#Exercici 7.
print("\nExercici 7.")
im=io.imread("Images/0019.png") #Inicialitzem im a la imatge
k_m=km.KMeans(im,3)
print(k_m.centroids)


#Exercici 8.
print("\nExercici 8.")
im=io.imread("Images/0019.png") #Inicialitzem im a la imatge
im=color.rgb2lab(im)
print(im[59][93]) # -1


#Exercici 9.
print("\nExercici 9.")
im=io.imread("Images/0019.png") #Inicialitzem im a la imatge
im=cn.ImColorNamingTSELabDescriptor(im)

print(im[48][2])




