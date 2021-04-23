import numpy as np
import pandas as pd

#Function used for min-max scaling
def min_max(x,x_max,x_min):
    x_scale = float((x-x_min)/(x_max-x_min))
    return x_scale
#Function used for Z-score Normalization
def Z_score(x,mean,std_dev):
    z_score = float((x-mean)/std_dev)
    return z_score

dataset =pd.read_csv("winequality-red.csv")   #Read data using pandas
L = len(dataset)                              #L is no.of training examples
att = dataset.iloc[:,:-1].values
d1 = np.zeros([L,12] , dtype =float)
d2 = np.zeros([L,12] , dtype =float)

qual_1 = dataset.iloc[:,11].values   #Assigning values of quality in dataset to qual_1
qual_A =np.zeros(L)                  #Initialization of 2 numpy arrays
qual_B =np.zeros(L)

for i in range(L):
    if qual_1[i] <= 6:
        qual_A[i] = 0
    elif qual_1[i] > 6:
        qual_A[i] = 1

qual_2 = dataset.iloc[:,11].values
for i in range(L):
    if (qual_2[i] < 5):
        qual_B[i] = 0
    elif(qual_2[i]==5 or qual_2[i]==6):
        qual_B[i] = 1
    else:
        qual_B[i] = 2
#Forming datasetA using min-max scaling function
for j in range(L):
    for k in range(12):
        if (k<=10):
           d1[j][k] = min_max(att[j][k],max(dataset.iloc[:,k].values),
                                       min(dataset.iloc[:,k].values))
        if k==11:
           d1[j][k] = qual_A[j]
#Forming datasetB using Z-score normalization function
for j in range(L):
    for k in range(12):
        if (k<=10):
          d2[j][k] = Z_score(att[j][k],np.mean(dataset.iloc[:,k].values),
                                      np.std(dataset.iloc[:,k].values))
        if k==11:
          d2[j][k] = qual_B[j]

min_arr = np.amin(d2,axis=0)
max_arr = np.amax(d2,axis=0)
#Function to assign value of datasetB to Bins
def interval(min,max,att_val):
    gap = float((min+max)/(4.0))
    temp1 = min+gap
    temp2 = temp1+gap
    temp3 = temp2+gap
    if(att_val>=min and att_val<temp1):
        bin = 0
    elif(att_val>=temp1 and att_val<temp2):
        bin = 1
    elif(att_val>=temp2 and att_val<temp3):
        bin = 2
    else:
        bin = 3
    return bin
#Assigning Bins
for i in range(11):
    for j in range(L):
        d2[j][i] = interval(min_arr[i],max_arr[i],d2[j][i])

d_A = np.zeros([L,12] , dtype =float)
d_B = np.zeros([L,12] , dtype =float)
def datasetA():
     d_A = d1
     return d_A
def datasetB():
     d_B = d2
     return d_B


def main():
 print("dataset_A :")
 print(d1)
 print("---------------------------------------------------------------------")
 print("dataset_B :")
 print(d2)

if __name__ == '__main__':
      main()