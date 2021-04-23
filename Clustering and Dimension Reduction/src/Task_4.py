import numpy as np
from random import *
from sklearn.decomposition import PCA
from Task_1 import TF_IDF
from Task_2 import AgglomerativeClustering
from Task_3 import kmeans

Vectors = TF_IDF()

pca = PCA(3)
pca.fit(Vectors)
B = pca.transform(Vectors)
centroid_arr = np.zeros([8,3],dtype = float)
#Generating 8-random centroids
for i in range(8):
    for j in range(3):
        centroid_arr[i][j] = random()
def main():
    print("Agglometrive clusters with reduced dimensions:")
    for i in range(8):
          print(AgglomerativeClustering(B,8)[i])

def Agglometrive_reduced():
    return AgglomerativeClustering(B, 8)

if __name__ == '__main__':
       main()
