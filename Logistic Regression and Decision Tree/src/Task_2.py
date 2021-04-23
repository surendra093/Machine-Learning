import numpy as np
import pandas as pd
from Task_1 import datasetA                #importing datasetA created in Task1
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
#function for Cost Function
def cost_fun(Hyp,y,len):
    H1 = np.log(Hyp)
    y1 = y
    j1 = np.dot(H1,y1)
    unit_mat1 = np.ones([L,1],dtype=float)
    unit_mat2 = np.ones([1, L], dtype=float)
    y2 = np.subtract(unit_mat1,y)
    H2 = np.log(np.subtract(unit_mat2,Hyp))
    j2 = np.dot(H2,y2)
    j = float((j1+j2)/(-len))
    return j

D1 = datasetA()

L = len(D1)
Y = np.zeros(L,dtype = float)          #initializing numpy array which stores values of classes
#x = np.zeros([12,L],dtype=float)       #numpy array  to store attribte values
hyp = np.zeros([1,L],dtype=float)      #numpy array to store hypothesis values
y = Y.reshape(L,1)
unit_arr = np.ones([1,L],dtype=float)
for i in range(L):                     #assigning values of quality from datasetA to 'y'
     y[i] = D1[i][11]

theta = np.ones(12)
a = theta.reshape(1,12)
def x_matrix(D1,L):                      #To form matrix x
   x = np.zeros([12,L],dtype=float)
   for i in range(L):
      x[0][i] = 1.0
   for i in range(L):
      for j in range(11):
          x[j+1][i] = D1[i][j]
   return x

x = x_matrix(D1,L)
alpha = 0.05                   #learning rate for gradient descent
const = alpha/L
def hyp_fun(a,x,L):          #Function to calculate hypothesis values
    temp = np.dot(a,x)
    exp = np.exp(-temp)
    for i in range(L):
        hyp[0][i] = (1.0/(1.0+exp[0][i]))
    return hyp

def error(a,x,L):
    err = np.subtract(hyp_fun(a, x, L), y.reshape(1, L))
    return err

def upd_theta(err,x,len):       #Gradient decent i.e updating values of parameters
  for j in range(12):
    sum = 0
    for i in range(L):
         sum = sum+(err[0][i])*(x[j][i])
    diff = const * sum
    a[0][j] = diff
  return a

def Logistic_reg(err,x,L):
   for k in range(2500):          #Minimizing the cost function value.
     a = upd_theta(err,x,L)
     #err = np.subtract(hyp_fun(a, x, L), y.reshape(1, L))
     err = error(a,x,L)
     #cost = cost_fun(hyp_fun(a,x,L),y,L)
   return a

def test(a,x,y,L):
   y_pre = np.zeros(L)        #array to store predicted class
   threshold = 0.51           #setting threshold to classify the two classes
   for i in range(L):
      if(hyp_fun(a,x,L)[0][i]>threshold):
          y_pre[i] = 1.0
      if(hyp_fun(a,x,L)[0][i]<=threshold):
          y_pre[i] = 0.0

   match = 0
   for j in range(L):
     if(y[j] == y_pre[j]):
          match = match+1
   #print("no.of datasets matched:",match)        #No.of predicted classes matched with actual classes
   y_predicted = np.array(y_pre).tolist()
   y_actual = np.array(y).tolist()
   accuracy = (match/L)
   '''
   confusion_matrix1 = confusion_matrix(y_actual, y_predicted)
   precision = (confusion_matrix1[0][0]) / (confusion_matrix1[0][0] + confusion_matrix1[1][0])
   recall = (confusion_matrix1[0][0]) / (confusion_matrix1[0][0] + confusion_matrix1[0][1])
   '''
   precision = precision_score(y_actual, y_predicted, average='macro')
   recall = recall_score(y_actual, y_predicted, average='macro')
   return accuracy,precision,recall

err = error(a, x, L)
print("Accuracy :"+str(test(Logistic_reg(err, x, L), x, y, L)[0]*100) +"%")    #To print Accuracy of model
'''
__________________Now using scikit-learn package, using saga solver____________________
'''
x_s = D1[:,:-1]
y_s = D1[:,11]
#Calculation of accuracy using scikit learn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(x_s, y_s, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
from sklearn.model_selection import cross_validate
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
y_pred = logmodel.predict(x_test)
print("Accuracy using scikit-learn :"+str((logmodel.score(x_test, y_test))*100)+"%")

'''
________________________Cross Validation_______________________________
'''
total_size = np.shape(D1)[0]         #total no.of training examples
set_div = int(total_size/3)          #To divide total set into 3 parts
break_points = [0, set_div, 2*set_div, total_size]

acc_sum = 0            #intializing acc,pre and recall to zero(to use for classifier without scikit learn)
pre_sum = 0
rec_sum = 0
accuracy_sum = 0       #intializing acc,pre and recall to zero(to use for classifier with scikit learn)
precision_sum = 0
recall_sum = 0

for i in range(3):                 #Separating the data into train data and test data for CV

    test_data = D1[break_points[i]:break_points[i+1],:]
    test_size = set_div

    m = total_size - test_size
    train_data = np.zeros((m,np.shape(D1)[1]))
    train_data[0:break_points[i],:] = D1[0:break_points[i],:]
    train_data[break_points[i]:,:] = D1[break_points[i+1]:,:]

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    '''
    cross validation for classifier without scikit learn
    '''
    x_train_matrix = x_matrix(train_data,m)
    x_test_matrix = x_matrix(test_data,test_size)
    fun = test(Logistic_reg(err,x,L),x_test_matrix,y_test,test_size)
    acc_sum += fun[0]
    pre_sum += fun[1]
    rec_sum += fun[2]
    '''
    cross validation for classifier with scikit learn
    '''
    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)
    y_pred = logmodel.predict(x_test)
    accuracy_sum = accuracy_sum+logmodel.score(x_test, y_test)
    precision_sum += precision_score(y_test, y_pred, average='macro')
    recall_sum += recall_score(y_test, y_pred, average='macro')

mean_acc  = (acc_sum/3)*100
mean_pre  = (pre_sum/3)*100
mean_rec  = (rec_sum/3)*100
mean_accuracy = (accuracy_sum/3)*100
mean_precision = (precision_sum/3)*100
mean_recall = (recall_sum/3)*100
print("_________________After Cross_Validation_______________")

print("mean accuracy :"+str(mean_acc)+"%")
print("mean precision :"+str(mean_pre)+"%")
print("mean recall :"+str(mean_rec)+"%")
print("mean accuracy_scikit :"+str(mean_accuracy)+"%")
print("mean precision_scikit:"+str(mean_precision)+"%")
print("mean recall_scikit :"+str(mean_recall)+"%")


