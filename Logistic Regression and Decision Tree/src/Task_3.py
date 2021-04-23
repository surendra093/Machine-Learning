import numpy as np
import pandas as pd
from Task_1 import datasetB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import  metrics


#Function to find entropy of node
def entropy(y):
    length = len(y)
    p_0 = np.count_nonzero(y == 0)/length
    p_1 = np.count_nonzero(y == 1)/length
    p_2 = np.count_nonzero(y == 2)/length

    if (p_0 == 0):
        E0 = 0
    else:
        E0 = -p_0 * np.log2(p_0)
    if (p_1 == 0):
        E1 = 0
    else:
        E1 = -p_1 * np.log2(p_1)
    if (p_2 == 0):
        E2 = 0
    else:
        E2 = -p_2 * np.log2(p_2)
    entropy = E0 + E1 + E2
    return entropy

class Decisiontree:
      def __init__(self):
           self.child_nodes = []
           self.label = 'none'
           self.type = -1

      def build_tree(self, x_matrix, y, default):
          # clear the old tree if the same instance is called again

          self.__init__()
          L = len(y)
          if (L < 10):
              if (L == 0):
                  self.type = default
                  return
              N_0 = np.count_nonzero(y == 0)
              N_1 = np.count_nonzero(y == 1)
              N_2 = np.count_nonzero(y == 2)
              if N_0 == max(N_0, N_1, N_2):
                  self.type = 0
              elif N_1 == max(N_0, N_1, N_2):
                  self.type = 1
              else:
                  self.type = 2
              return
          if (entropy(y) == 0):
              self.type = y[0]
              return

          n = np.shape(x_matrix)[1]
          min_entropy = float('inf')
          min_attribute = -1
          for i in range(n):
              x = x_matrix[:, i]
              ent = 0
              if (x[0] == -1):
                  continue

              # for each value the attribute i can take, partition it
              for val in range(4):
                  count_val = np.count_nonzero(x == val)
                  y_val = [y[j] for j in range(L) if x[j] == val]
                  fraction_val = count_val / L

                  # if empty partition, its contribution to entropy is 0
                  if count_val == 0:
                      temp = 1
                  else:
                      temp = entropy(y_val)

                  ent += fraction_val * temp

              if (ent < min_entropy):
                  min_entropy = ent
                  min_attribute = i

          # if no attributes are left to partition, assign the majority class

          if (min_attribute == -1):
              count0 = np.count_nonzero(y == 0)
              count1 = np.count_nonzero(y == 1)
              count2 = np.count_nonzero(y == 2)

              if count0 == max(count0, count1, count2):
                  self.type = 0
              elif count1 == max(count0, count1, count2):
                  self.type = 1
              else:
                  self.type = 2
              return

          self.label = min_attribute

          for val in range(4):
              x_matrix_val = np.array([x_matrix[i, :] for i in range(L) if x_matrix[i, min_attribute] == val])
              if (x_matrix_val.size > 0):
                  x_matrix_val[:, min_attribute] = -1

              y_val = np.array([y[i] for i in range(L) if x_matrix[i, min_attribute] == val])

              node = Decisiontree()
              self.child_nodes.append(node)
              node.build_tree(x_matrix_val, y_val, default)

      def predict_row(self, x):
          if (self.label == 'none'):
              return self.type

          val = int(x[self.label])
          return self.child_nodes[val].predict_row(x)

      def predict(self, x_matrix):
          y = []
          for x in x_matrix:
              if (self.label == 'none'):
                  y.append(self.type)
              else:
                  y.append(self.predict_row(x))

          return y

D2 = datasetB()
X = D2[:,:-1]
Y = D2[:,11]
print("For model using scikit  learn :")
#Scikit learn model using complete dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
clf_entropy = DecisionTreeClassifier( criterion = "entropy", min_samples_split = 10,random_state=1)
clf_entropy.fit(x_train, y_train)
y_pred = clf_entropy.predict(x_test)

report = metrics.classification_report(y_test, y_pred)
print("Report :",report)
print ("Accuracy : "+str(accuracy_score(y_test,y_pred)*100)+"%")
print("-------------------------------------------------------------")
print("________________After Cross Validation_______________________")
total_size = np.shape(D2)[0]         #total no.of training examples
set_div = int(total_size/3)          #To divide total set into 3 parts
break_points = [0, set_div, 2*set_div, total_size]

acc_sum =0              #intializing acc,pre and recall to zero(to use for classifier with scikit learn)
pre_sum =0
rec_sum =0
accuracy_sum = 0        #intializing acc,pre and recall to zero(to use for classifier without scikit learn)
precision_sum = 0
recall_sum = 0

for i in range(3):
    # Separating the data into train data and test data for CV
    test_data = D2[break_points[i]:break_points[i+1],:]
    test_size = set_div

    m = total_size - test_size
    train_data = np.zeros((m,np.shape(D2)[1]))
    train_data[0:break_points[i],:] = D2[0:break_points[i],:]
    train_data[break_points[i]:,:] = D2[break_points[i+1]:,:]

    x_matrix = train_data[:,:-1]
    x_matrix = x_matrix.astype(int)
    y = train_data[:,-1]
    y = y.astype(int)

    default = np.argmax(np.bincount(y))
    # Decision tree model without scikit learn
    dec_tree = Decisiontree()
    dec_tree.build_tree(x_matrix,y,default)

    test_matrix = test_data[:,:-1]
    test_y = test_data[:,-1]

    predicted = np.array(dec_tree.predict(test_matrix))
    accuracy_sum += accuracy_score(test_y, predicted) * 100
    precision_sum += precision_score(test_y, predicted, average='macro')
    recall_sum += recall_score(test_y, predicted, average='macro')
    #Decision tree model using scikit learn
    clf_entropy = DecisionTreeClassifier(criterion="entropy", min_samples_split=10, random_state=1)
    clf_entropy.fit(x_matrix, y)
    y_pred = clf_entropy.predict(test_matrix)
    acc_sum += accuracy_score(test_y, y_pred) * 100
    pre_sum += precision_score(test_y, y_pred,average='macro')
    rec_sum += recall_score(test_y, y_pred,average='macro')
    report = metrics.classification_report(test_y, y_pred)
    print("Report for "+str(i+1)+" validation set : ")
    print(report)
#mean values for classifier without using scikit learn
mean_accuracy = (accuracy_sum)/3
mean_precision = (precision_sum*100)/3
mean_recall = (recall_sum*100)/3
#mean values for classifier using scikit learn
mean_acc = (acc_sum)/3
mean_pre = (pre_sum*100)/3
mean_rec = (rec_sum*100)/3

print("-------------------------------------------------------------")
print("For model without using scikit learn :")
print('mean accuracy = '+str(mean_accuracy)+'%')
print('mean precision = '+str(mean_precision)+'%')
print('mean recall = '+str(mean_recall)+'%')
print("-------------------------------------------------------------")
print("For model using scikit learn :")
print("mean accuracy = "+str(mean_acc)+"%")
print("mean precision = "+str(mean_pre)+"%")
print("mean recall = "+str(mean_rec)+"%")


