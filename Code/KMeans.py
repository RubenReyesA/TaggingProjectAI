
"""

@author: ramon, bojana
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
import math

def NIUs():
    return 1382357, 1495036

def auto_K(fisher):
    
    """
    Aquesta funció calcula de manera automàtica quina és la millor K, i la retorna.
    """
    
    max_fisher=[]
    max_dif=[]
    
    for index, value in enumerate(fisher[1:]):
        index+=1
        value_ant=index-1
        dif=abs(fisher[value_ant]-value)
        max_fisher.append(dif)
    
    for index, value in enumerate(max_fisher[1:]):
        index+=1
        value_ant=index-1
        dif=max_fisher[value_ant]/value
        max_dif.append(dif)
        
    maximum=max(max_dif)
    
    maximum_index=max_dif.index(maximum)
    
    maximum_index+=3
    
    return maximum_index
    
    
def dist_fitting(a,b):
    """
    Aquesta funció es utilitzada per calcular el discrimant de Fisher. Concretament,
    calcula la distància.
    """
    return np.sqrt(np.sum((a-b)**2))

def plot_results(axisx,axisy,best_k,value_k):
    
    """
    Aquesta funció ha sigut creada per mostrar la gràfica dels valors de K.
    Aquest valor apareixen representats en una recta i els valors marcats
    amb punts vermells. El valor de K escollit com millor K es marca com amb una
    X de color verd.
    """
    plt.plot(axisx, axisy, 'ro-', markersize=8, lw=2)
    plt.plot(best_k,value_k, 'gX-',markersize=18, lw=2)
    plt.grid(True)
    plt.xlabel('Num Clusters (K)')
    plt.ylabel('Fisher Value')
    plt.show()
    
def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
    

    """ Calculem la diferencia entre els arrays X i C per columnes,
        i guardem els valors a un nou array que és el que retornem. """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
  
    distance_array = np.zeros((X.shape[0], C.shape[0])) # Crea la matriu resultant amb zeros.
    distance_temp_row=np.zeros((X.shape[0],X.shape[1])) # Crea la matriu temporal amb zeros.
    
    # Per cada centroid, calcula la diferencia entre els punts.
    # Per fer això és fa la resta, s'eleva al quadrat, es realitza la suma de les dimensions,
    # i el resultat es fa l'arrel quadrada. I el resultat equival a una columna de la matriu final.
    # Hi ha tantes columnes en distance com centroids té C.
    
    for centroid in range(C.shape[0]):
        distance_temp_row=X-C[centroid]
        distance_temp_row=distance_temp_row**2
        distance_temp_row=distance_temp_row.sum(axis=1)
        distance_temp_row=np.reshape(distance_temp_row,(distance_temp_row.shape[0],1))
        distance_temp_row=np.sqrt(distance_temp_row)
        distance_array[:,[centroid]]=distance_temp_row
        
    return distance_array

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """
        self._init_original_image(X)
        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

    def _init_original_image(self,X):
        self.original_image=X
        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
        
        
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
       
        """ Si la dimensió de la matriu és 3, la transformem en 2.
            On les files són tots els punts de la imatge
            i les columnes són els valors d'aquest punt en les dimensions.
            En cas que la dimensió sigui diferent, l'assignem directament """
            
        if (len(X.shape) == 3):
            self.X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
        else:
            self.X = X
            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first' # Editar para el exercici 7
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'fisher'


        self.options = options
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K                                             # INT number of clusters
        if (self.K>0):
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        else:
            self.bestK()
        self.num_iter = 0                                      # INT current iteration
        
            
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################


    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        """ Inicialitzem l'array de centroids com els primers píxels diferents
            de la imatge, en cas que l'opció escollida sigui "first", en cas
            que l'opció escollida sigui "random" inicialitzem l'array com K pixels
            de la imatge escollits de manera aleatòria. """
            
        if self.options['km_init'].lower() == 'first':

            centroid_list = [] #Llista on anirem afegint tots els centroids, que després convertirem a np.array.
            n_centroids = 0 # Contador del nombre de centroids afegits, que haurà de ser igual a K.
            i = 0 # Índex on s'ha d'afegir un determinat centroid.
            
            while (n_centroids < self.K):
                found = -1
                for centroid in centroid_list:
                    # Comprova si el centroid de la llista és igual al que es troba en la mateixa posició de l'array X, per totes les columnes.
                    
                    if ((centroid == self.X[i]).all()): 
                        found = 1  
                
                if (found==-1):
                    centroid_list.append(self.X[i]) # Afegim a la llista el punt de la imatge que ara és centroid.
                    n_centroids+=1
                i+=1
            
            self.centroids = np.array(centroid_list) # Convertim la llista a un array i el guardem a self.centroids
      
        else:
            self.centroids = np.random.rand(self.K,self.X.shape[1]) # En cas que sigui random..
        self.centroids=self.centroids.astype(float)  # Guardem les dades en tipus float perquè funcioni correctament el programa.
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X.
        """
        
        """ Calculem l'array de distancies entre els píxels de X i les píxels dels centroids.
            Després guardem a clústers l'array d'índexs dels mínims d'aquestes distancies per files"""
            
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        distanciapixeltocentroid = distance(self.X, self.centroids) # Calculem l'array de distàncies.
        self.clusters = np.argmin(distanciapixeltocentroid,axis=1) # Gràcies a la llibreria np.array, definim els índexs dels mínims per files.
        
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        
        """ Recalculem els nous punts centroids. Assignem els self.old_centroids, buidem els
            self.centroids i després, agrupem per cada centroid els clústers, i després realitzem
            la mitjana per columnes per calcular les noves posicions dels píxels dels centroides. """
           
     
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        
        self.old_centroids = self.centroids # Assignació a self.old_centroids
        self.centroids=np.zeros((self.old_centroids.shape[0],self.old_centroids.shape[1])) # Buidatge dels self.centroids.
         
        for i in range(len(self.centroids)):
            points_of_centroid_i=self.X[self.clusters==i] # Filtratge a llista de tots els punts que tenen el mateix centroid i.
            
            if len(points_of_centroid_i)>0:
                mean_columns_points=np.mean(points_of_centroid_i, axis=0)  #Mitjana de l'array per columnes.
                self.centroids[i]=mean_columns_points #Assignació del nou centroid.
        

         
    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        
        """ Realitzem un nou array que contingui les distàncies entre els centroid d'aquesta
            iteració i de la iteració anterior. Després recorrem l'array sencer, i anem comparant
            si algún valor supera la tolerància màxima definida a les opciones del K-Means. """
           
        
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        distance_between_centroids = distance(self.centroids, self.old_centroids) # Array de les distàncies entre els centroid d'aquesta iteracío i de la iteració anterior.

        for i in range (distance_between_centroids.shape[0]):
            if (distance_between_centroids[i][i] > self.options['tolerance']): # Comprovació que cap valor superi el valor "tolerance".
                return False
        
        
        return True
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        
        """
        Itera una vegada el algorisme K-Means
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """  
        """
        Executa el K-Means fins que convergi
        """
        
        if (self.K==0):
            self.bestK()
            return
            

        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)
        
        if self.options['verbose']:
            self.show_image()
            self.plot()
            
    def bestK(self):
        """
        Aquesta funció s'encarrega d'executar el bestK, determinar quina és la
        millor K, fent crides a funcions anteriors i executar l'algorisme per
        la millor K. S'utlitzen les funcions plot_results i autoK.
        """
        
        fisher_list=[]
        
        for loop in range(2,7):
            self._init_rest(loop)
            self.run()
            fisher_list.append(self.fitting())
            
        
        best_K=auto_K(fisher_list)
        axisx=[2,3,4,5,6]
        index=axisx.index(best_K)
        value=fisher_list[index]
        
        if self.options['verbose']:
            plot_results(axisx,fisher_list,best_K,value)
        
        self.K=best_K
        
        if self.options['verbose']:
            plt.figure(1)
            plt.imshow(self.original_image)
            plt.axis('off')
        
        self._init_rest(self.K)
        self.run()
        
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        
        """
        Calcula el valor del discriminant de Fisher de una determinada execució
        de l'algorisme K-Means. Retorna aquest valor. Utilitza la funció dist_fitting
        per calcular la distància.
        """

        if self.options['fitting'].lower() == 'fisher':
            
            intra_class_value=0
            inter_class_value=0
            fisher=0
            
            for k in range(self.K):
                valor=0
                nk=(np.sum(self.clusters==k))
                
                for i in range(nk):
                    valor+=dist_fitting((self.X[self.clusters==k,:][i,:]),(self.centroids[k,:]))
                
                valor=valor/nk
                intra_class_value+=valor
            
            intra_class_value=intra_class_value/self.K
            
            for k2 in range(self.K):
                inter_class_value+=dist_fitting((self.centroids[k2,:]),(np.mean(self.X,axis=0)))
                
            inter_class_value=inter_class_value/self.K
            
            fisher=intra_class_value/inter_class_value
            return fisher
            
        else:
            return np.random.rand(1)
        

    def plot(self, first_time=True):
        """
            Aquesta funció és l'encarregada de mostrar de forma gràfica amb els
            gràfics els resultats de l'execució. Aquesta funció havia estat dissenyada
            pels professors però ha estat modificada gràcies a un codi penjat al
            Caronte que millorava la versió i feia els resultats molt més intuitius.

        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=self.X / 255, marker='.', alpha=0.3)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], c=self.centroids / 255, marker='o', s=5000, alpha=0.75, linewidths=1, edgecolors="k")

        textdict = "K: "+str(self.K)+"\nInit: "+str(self.options["km_init"]+"\nIter: "+str(self.num_iter))
        box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text( 200, 100, 200, textdict, transform=ax.transAxes, fontsize=16, horizontalalignment="left", verticalalignment="bottom", bbox=box)

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        
        plt.show()
        
    def show_image(self):
        """
            Aquesta funció mostra la imatge real de cada iteració, aplicant els
            respectius filtres de color. Aquesta funció, com l'anterior, van ser
            penjades al Caronte per tal de millor l'anàlisi del resultats.
        """

        if self.num_iter == 0:
            plt.imshow(self.original_image)
        else:
            new_image = np.copy(self.X)
            for ctr in range(self.K):
                if new_image[self.clusters == ctr].shape[0] > 0:
                    new_image[self.clusters == ctr] = self.centroids[ctr]
            plt.imshow(new_image.reshape(self.original_image.shape))
            plt.show()
            plt.pause(1)