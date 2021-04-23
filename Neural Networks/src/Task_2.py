import numpy as np
from sklearn.neural_network import MLPClassifier
from Dataset import train_dataset
from Dataset import test_dataset

def preprocess():
    x_train,y_train = train_dataset()
    x_test ,y_test = test_dataset()
    return x_train,y_train,x_test,y_test

def one_hot_vect(y):
    vect_arr = np.zeros([y.shape[0],3])
    for i in range(y.shape[0]):
        if y[i][0]==1 :
            vect_arr[i][0] = 1
            vect_arr[i][1] = 0
            vect_arr[i][2] = 0
        if y[i][0]==2 :
            vect_arr[i][0] = 0
            vect_arr[i][1] = 1
            vect_arr[i][2] = 0
        if y[i][0]==3 :
            vect_arr[i][0] = 0
            vect_arr[i][1] = 0
            vect_arr[i][2] = 1
    return vect_arr

# function to find accuracy
def accuracy(predicted_class,actual_class):
    m = actual_class.shape[0]
    #predicted_class = con_prob_to_class(outputs)
    sum = 0
    for i in range(m):
        if np.array_equal(predicted_class[i],actual_class[i]):
            sum = sum+1
    accuracy = (sum/m)*100
    return accuracy

X_train,y_train,X_test,y_test = preprocess()
Y_train = np.reshape(y_train,(len(y_train),1))
Y_test = np.reshape(y_test,(len(y_test),1))

Y1_train = one_hot_vect(Y_train)
Y1_test = one_hot_vect(Y_test)

classifier1 = MLPClassifier(hidden_layer_sizes=(32),activation="logistic",solver='sgd',batch_size=32,
                            learning_rate="constant",learning_rate_init=0.01,max_iter=200,random_state=10)

classifier1.fit(X_train,Y1_train)
#Predicting y for X_test and X_train
y_train_pred = classifier1.predict(X_train)
y_test_pred = classifier1.predict(X_test)

print("Part 2 Specification 1A:")
print("Training accuracy of Task_1A :"+str(accuracy(y_train_pred,Y1_train))+"%")
print("Test accuracy of Task_1A :"+str(accuracy(y_test_pred,Y1_test))+"%")
print("------------------------------------------------------")

classifier2 = MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",solver='sgd',batch_size=32,
                            learning_rate="constant",learning_rate_init=0.01,max_iter=200)

classifier2.fit(X_train,Y1_train)
y_train_pred = classifier2.predict(X_train)
y_test_pred = classifier2.predict(X_test)

print("Part 2 Specification 1B :")
print("Training accuracy of Task_1B :"+str(accuracy(y_train_pred,Y1_train))+"%")
print("Test accuracy of Task_1B :"+str(accuracy(y_test_pred,Y1_test))+"%")