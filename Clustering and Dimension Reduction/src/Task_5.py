import numpy as np
import pandas as pd
from Task_2 import Final_clusters
from Task_3 import final_clusters
from Task_4 import Agglometrive_reduced

Task2_clusters = Final_clusters()
Task3_clusters = final_clusters()
Task4_agglo_Rc = Agglometrive_reduced()

#Function to find entropy
def entropy(y):
    length = len(y)
    p = np.zeros([8,1])
    E = np.zeros([8,1])
    for i in range(8):
        p[i] = np.count_nonzero(y == i)/length

    for j in range(8):
        if(p[j] == 0):
             E[j] = 0
        else:
            E[j] = -p[j] * np.log2(p[j])

    entropy = np.sum(E)
    return entropy
def cumm_len(task):
    sum = np.zeros([8,1],dtype=int)
    for i in range(8):
        sum[i] = sum[i-1]
        sum[i] = sum[i]+int(len(task[i]))
    return sum
def cumm_sum(y):
    sum = np.zeros([8, 1], dtype=int)
    for i in range(8):
        sum[i] = sum[i - 1]
        sum[i] = sum[i] + np.count_nonzero(y==i)
    return sum

def cond_prob(cluster,Y):
     p = np.zeros([8,1])
     length = len(cluster)
     for i in range(8):
         sum = 0
         for element in cluster:
             if i==0:
                 if (0<=element<cumm_sum(Y)[i][0]):
                     sum = sum+1
             else:
                 if (cumm_sum(Y)[i-1][0]<= element < cumm_sum(Y)[i][0]):
                     sum = sum +1
             p[i] = sum / length
     return p

def cond_entropy(cluster,Y):
    E = np.zeros([8, 1])
    for i in range(8):
       if cond_prob(cluster,Y)[i] == 0:
           E[i] = 0
       else:
           E[i] = cond_prob(cluster,Y)[i] * np.log2(cond_prob(cluster,Y)[i])
    sum = np.sum(E)
    entropy = - (len(cluster)/Y.shape[0]) * sum
    return entropy

def mutual_information(task,Y):
      E_class = entropy(Y)
      sum = 0
      for i in range(8):
          sum = sum + cond_entropy(task[i],Y)
      E_class_incluster = sum
      mut_info = E_class - E_class_incluster
      return mut_info

def normal_mutual_info(task):
    dataset = pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")
    D = dataset.iloc[:,:]
    class_labels = D.class_label
    rows = class_labels.shape[0]
    Y = np.zeros([rows,1],dtype=int)      #class labels
    y = np.zeros([rows,1],dtype=int)      #predicted clusters
    for i in range(rows):
        if (class_labels[i]=="Buddhism"):
                 Y[i] = 0
        if (class_labels[i]=="TaoTeChing"):
                 Y[i] = 1
        if (class_labels[i]=="Upanishad"):
                  Y[i] = 2
        if (class_labels[i]=="YogaSutra"):
                  Y[i] = 3
        if (class_labels[i]=="BookOfProverb"):
                  Y[i] = 4
        if (class_labels[i]=="BookOfEcclesiastes"):
                  Y[i] = 5
        if (class_labels[i]=="BookOfEccleasiasticus"):
                 Y[i] = 6
        if (class_labels[i]=="BookOfWisdom"):
                 Y[i] = 7
    for  j in range(8):
      if (j == 0):
         for  k in range(len(task[j])):
              y[k] = j
      else:
          for k in range(cumm_len(task)[j-1][0],cumm_len(task)[j][0]):
              y[k] = j

    E_class = entropy(Y)
    E_cluster = entropy(y)
    I = mutual_information(task,Y)
    NMI_score = (2*I)/(E_class+E_cluster)

    return NMI_score

print("Task2 NMI score:"+str(normal_mutual_info(Task2_clusters)))
print("Task3 NMI score:"+str(normal_mutual_info(Task3_clusters)))
print("Task4 Agglometrive reduced clusters score:"+str(normal_mutual_info(Task4_agglo_Rc)))