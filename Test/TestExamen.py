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
    
options={'verbose':False,'km_init':'first'}


#Dades inicials dels Punts i Centroides
X=np.array([[239,44,35],[179,267,254],[175,158,1],[204,21,32],[284,205,239],[265,58,279]])
C=np.array([[211.0,116.0,22.0],[202.0,13.0,50.0]])



#Exercici 1. Distancia euclidiana del punt 6 a tots els centroides
print("\nExercici 1.")
d=km.distance(X,C)
print(d)


#Exercici 3. Si iterem l'algorisme KMeans 1 cop a partir de les dades que tenim, quins centroides obtindriem?
#Treure el codi del getCentroids
#MALAMENT EN EL GITANO
print("\nExercici 3.")
X=np.array([[239,44,35],[179,267,254],[175,158,1],[204,21,32],[284,205,239],[265,58,279]])
options={'verbose':False,'km_init':'first'}
k_m=km.KMeans(X,2,options)
print(k_m._init_X)
k_m.centroids=np.array([[211.0,116.0,22.0],[202.0,13.0,50.0]])
k_m._get_centroids()
print(k_m.centroids)

#Exercici 6. Donada la imatge 0065.png quins centroides obtindrem si el mètode d'inicialització és 'first' amb K=3
# en l'espai RGB? Nota: l'ordre en que apareixen els centroides no és rellevant.
print("\nExercici 6.")
im=io.imread("Images/0065.png") #Inicialitzem im a la imatge
options={'verbose':False,'km_init':'first'}
k_m=km.KMeans(im,3,options)
k_m.run()
print("Original:")
print(k_m.centroids)


#Exercici 7. Considerant K=3 i si afegim una nova opció en el mètode de inicialització que anomenarem 'custom'
# que executa el se¨guent codi:
print("\nExercici 7.")
exe_7=km.KMeans(im,3,options)
print(exe_7.centroids)


#Exercici 8
print("\nExercici 8.")
im=io.imread("Images/0065.png") #Inicialitzem im a la imatge
im=color.rgb2lab(im)
print(im[41][31])


#Exercici 9
print("\nExercici 9.")
im=io.imread("Images/0065.png") #Inicialitzem im a la imatge
im=cn.ImColorNamingTSELabDescriptor(im)
print(im[17][65])






