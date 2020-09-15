# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
import time

if os.path.isfile('TeachersLabels.py') and True:
    import TeachersLabels as lb
else:
    import Labels as lb



plt.close("all")
if __name__ == "__main__":

    t = time.time()
    options = {'colorspace':'RGB', 'K':0, 'single_thr':0.25, 'verbose':True, 'km_init':'first', 'metric':'basic'}

    ImageFolder = 'Images'
    GTFile = 'LABELSlarge.txt'

    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []

    png_list=[]
    K_list=[]
    K_origin_list=[]
    Colour_list=[]
    Colour_Origin_list=[]
    similarity_list=[]

    for gt in GT:
        png_list.append(gt[0])
        K_origin_list.append(len(gt[1]))
        Colour_Origin_list.append(gt[1])
        print(gt[0])
        im = io.imread(ImageFolder+"/"+gt[0])
        colors,_,km = lb.processImage(im, options)
        K_list.append(km.K)
        Colour_list.append(colors)
        DBcolors.append(colors)

    encert,similarity_list = lb.evaluate(DBcolors, GT, options)
    print("Encert promig: "+ '%.2f' % (encert*100) + '%')
    print(time.time()-t)

    similarity_list=[str(x) for x in similarity_list]
    similarity_list = [c.replace('.', ',') for c in similarity_list]
    print(png_list,K_list,similarity_list)

    file=open("st02.txt","w")

    cadena="Nom de la imatge___"+"Valor K - GT___"+"Colours - GT___"+"Valor K - BestK___"+"Colours - BestK___"+"% Similitud"
    file.write(cadena+"\n\n")

    for png_name, K_GT, colourGT, K_value, colourK, similarity_value in zip(png_list,K_origin_list,Colour_Origin_list,K_list,Colour_list,similarity_list):

        cadena=png_name+"___"+str(K_GT)+"___"+str(colourGT)+"___"+str(K_value)+"___"+str(colourK)+"___"+str(similarity_value)
        file.write(cadena+"\n")


    file.close()
