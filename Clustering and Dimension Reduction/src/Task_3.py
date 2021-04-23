import numpy as np
import pandas as pd
from random import *
import math
from Task_1 import TF_IDF

Vectors = TF_IDF()
dataset = pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")
att = dataset.iloc[:,1:].values
columns = att.shape[1]
rows = att.shape[0]
k = 8 #number of clusters

centroid_arr = np.zeros([8,columns],dtype = float)
sim_matrix = np.zeros([rows,k],dtype = float)
dist_matrix = np.zeros([rows,k],dtype=float)
#Generating 8-random centroids
for i in range(k):
    for j in range(columns):
        centroid_arr[i][j] = random()

def similarity(centroid_mat,vectors):
    for l in range(k):
        for m in range(rows):
            #Similarity of "m"th vector with "l"th centroid
            sim_matrix[m][l] = np.dot(centroid_mat[l],vectors[m])
    return sim_matrix

def distance(sim_mat):
    for n in range(k):
        for o in range(rows):
            dist_matrix[o][n] = 1/(math.exp(sim_mat[o][n]))
    return dist_matrix

def mean_centroid(list):
    if (list==[]):
        mean_values = np.zeros([3])
    else:
        mean_values = np.mean(list,axis=0)
    return mean_values

def Sort(clusterlist):
    arr = []
    sh = len(clusterlist)
    for i in range(0, sh):
        arr.append(sorted(clusterlist[i]))

    arr1 = []
    for i in range(0, sh):
        x = arr[i]
        lis = [i, x[0]]
        arr1.append(lis)
    arr1.sort(key=lambda lis: lis[1])

    arr2 = []
    for i in range(0, sh):
        arr2.append(arr[arr1[i][0]])
    return arr2

def kmeans(Vectors,centroid_arr):
 for loop in range(1000):
   list1, list2, list3, list4, list5, list6, list7, list8 = ([] for i in range(8))
   List1, List2, List3, List4, List5, List6, List7, List8 = ([] for i in range(8))
   min_dist = np.amin(distance(similarity(centroid_arr,Vectors)),axis=1)
   min_indices = np.argmin(distance(similarity(centroid_arr,Vectors)),axis=1)
   rows = Vectors.shape[0]
   for i in range(rows):
        if (min_indices[i] == 0.0):
            list1.append(Vectors[i])
            List1.append(i)
        elif (min_indices[i]== 1.0):
            list2.append(Vectors[i])
            List2.append(i)
        elif (min_indices[i] == 2.0):
            list3.append(Vectors[i])
            List3.append(i)
        elif (min_indices[i] == 3.0):
            list4.append(Vectors[i])
            List4.append(i)
        elif (min_indices[i] == 4.0):
            list5.append(Vectors[i])
            List5.append(i)
        elif (min_indices[i] == 5.0):
            list6.append(Vectors[i])
            List6.append(i)
        elif (min_indices[i] == 6.0):
            list7.append(Vectors[i])
            List7.append(i)
        elif (min_indices[i] == 7.0):
            list8.append(Vectors[i])
            List8.append(i)
   centroid_arr[0] = mean_centroid(list1)
   centroid_arr[1] = mean_centroid(list2)
   centroid_arr[2] = mean_centroid(list3)
   centroid_arr[3] = mean_centroid(list4)
   centroid_arr[4] = mean_centroid(list5)
   centroid_arr[5] = mean_centroid(list6)
   centroid_arr[6] = mean_centroid(list7)
   centroid_arr[7] = mean_centroid(list8)

 clusters = []
 clusters.append(List1)
 clusters.append(List2)
 clusters.append(List3)
 clusters.append(List4)
 clusters.append(List5)
 clusters.append(List6)
 clusters.append(List7)
 clusters.append(List8)
 return Sort(clusters)

def final_clusters():
    return kmeans(Vectors,centroid_arr)

def main():
   print("______________All the eight clusters_______________________")
   C = kmeans(Vectors, centroid_arr)
   for cluster in range(k):
         print(C[cluster])

if __name__ == '__main__':
       main()