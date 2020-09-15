# -*- coding: utf-8 -*-
"""

@author: ramon, bojana
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1495036, 1382357

def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """
    
    """
    Realitza la lectura del GT
    """
    
    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """

    """
    Aquesta funció s'encarrega de fer l'avaluació completa de la nostra execució.
    Per cada imatge que hagi estat testejada amb el programa retorna el percentatge
    en tant per u, i com a valor retorna l'encert promig.
    """
    
    scores=[]
    for i in range(len(description)):
        scores.append(similarityMetric(description[i],GT[i][1],options))
    return sum(scores)/len(description), scores

def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    
    """
    Aquesta funció s'encarrega de calcular el nivell de similitud que hi ha 
    entre els resultats proporcionats pel nostre programa amb les solucions
    proporcionades en el Ground-Truth.
    
    Volem destacar que pensem que la equació proporcionada al guió del projecte
    no era correcta, perque la intersecció del nostres colors i els colors del GT
    són dividits per el length del nostre resultat, mentre que creiem que hauria 
    de ser el contrari. L'explicació explica de forma clara que ha de ser entre el GT,
    però l'equació diu el contrari. Nosaltres pensem que hauriem d'haver fet servir
    el que la explicació diu, però vam decidir mantenir l'equació intacta sense fer
    cap modificació. Aquesta pas està marcat amb un "Warning" a la mateixa línia.
    
    """
    
    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'
        
    if options['metric'].lower() == 'basic'.lower():
        inter=[]
        Est=set(Est)
        GT=set(GT)
        inter=Est & GT
        ret=float(len(inter))/float(len(Est)) #WARNING!!!
        return ret       
    else:
        return 0
        
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label

    """
    
    """
    Aquesta funció calcula a partir dels centroids rebuts pel K-Means les seves
    respectives etiquetes.
    """
    
    CentroidList=np.copy(kmeans.centroids)

    #En cada columna es marca el valor màxim de cada fila.
    MaxCentroidList=np.argmax(CentroidList,axis=1)
    
    Colours_values=[]
    Colours_indexes=[]
    
    for x in range(MaxCentroidList.shape[0]):
        # Per a cada centroid analitzem si ja arriba o supera el valor establert. Etiqueta simple.
        # Sino calculem ordenem el vector i seleccionem les dues etiquetes. Etiqueta composta.
        if (CentroidList[x][MaxCentroidList[x]] >= options['single_thr']):
            ColourValue=cn.colors[MaxCentroidList[x]]            
        else:
            vector_ordered=np.argsort(CentroidList[x])
            
            ColourAUX=[]
            ColourAUX.append(cn.colors[vector_ordered[-1]])
            
            ColourAUX.append(cn.colors[vector_ordered[-2]])
            
            ColourAUX.sort()
            
            ColourValue=ColourAUX[0]+ColourAUX[1]

        #Comprovem que el color no hagi sigut afegit abans. Això ho fem per evitar duplicitat d'etiquetes.
        if(ColourValue not in Colours_values):
               Colours_values.append(ColourValue)
               Colours_indexes.append([])           
        
        #Realitzem aquest procediment per afegir els indexos en cas que sigui possible.
        ColourArray=np.array(Colours_values)
        index=np.where(ColourValue==ColourArray)[0]
        Colours_indexes[index[0]].append(x)
            
    return Colours_values, Colours_indexes

   
def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """
    """
    Aquesta és la funció principal en la que es basa tot el projecte. 
    Aqui s'ha de transformar la imatge donada a un espai de tres dimensions per
    tal que es pugui treballar amb el nostre algorisme K-Means.
    Després es troben les etiquetes de color a partir dels centroids i es determina
    el percentatge d'èxit que ha tingut la nostra execució.
    """

##  1- Canviem l'espai de color donat per a que funcioni amb el nostre algorisme.
    if options['colorspace'].lower() == 'ColorNaming'.lower():
        im=cn.ImColorNamingTSELabDescriptor(im)
    
    elif options['colorspace'].lower() == 'RGB'.lower(): 
        pass

    elif options['colorspace'].lower() == 'Lab'.lower():        
        im=color.rgb2lab(im)

##  2- Analitzem el paràmetre K per determinar si hem o no d'executar el bestK o no.
##      Com que nosaltres fem la crida si i només si K=0, en cas que sigui K=1, executem bestK.
        
    if options['K']<2:
        kmeans = km.KMeans(im, 0, options)
    else:
        kmeans = km.KMeans(im, options['K'], options) 
    
    kmeans.run()

##  3- Obtenim les etiquetes dels colors de la nostra imatge.
    if options['colorspace'].lower() == 'Lab'.lower():    
        kmeans.centroids=color.lab2rgb([kmeans.centroids])[0]*255
        kmeans.centroids=cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
       
    elif options['colorspace'].lower() == 'RGB'.lower():
        kmeans.centroids=cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
        
   
#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)   
    return colors, which, kmeans