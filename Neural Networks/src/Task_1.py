import numpy as np
import random
import matplotlib.pyplot as plt
from Dataset import train_dataset
from Dataset import test_dataset

def sigmoid(x):
    fun = 1/(1+np.exp(-x))
    return fun

def ReLu(x):
    fun = np.maximum(0,x)
    return fun
#function to split in train and test set
def preprocess():
    x_train,y_train = train_dataset()
    x_test ,y_test = test_dataset()
    return x_train,y_train,x_test,y_test
#function to initialize weights
def Weight_initialiser(input,output):
    weight_arr = np.zeros([input,output])
    for i in range(input):
        for j in range(output):
            weight_arr[i,j] = random.uniform(-1,1)
    return weight_arr
#activation function of output layer
def softmax(X):
    probability_array = []
    for i in range(X.shape[0]):
      x = X[i]
      a = x[0]
      b = x[1]
      c = x[2]
      P1 = np.exp(a)/np.sum([np.exp(a),np.exp(b),np.exp(c)])
      P2 = np.exp(b)/np.sum([np.exp(a),np.exp(b),np.exp(c)])
      P3 = np.exp(c)/np.sum([np.exp(a),np.exp(b),np.exp(c)])
      probability_array.append(np.array([P1,P2,P3]))
    return probability_array
#function to create mini_batches
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    n_minibatches = data.shape[0] // batch_size
    i = 0
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches
#function to convert class into one_hot_vector
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

def single_layer_forward(x_prev,bias,w_curr,activation_fun):
     z_curr = np.dot(x_prev,w_curr)+bias
     global x_current
     if activation_fun is "sigmoid":
         x_current = sigmoid(z_curr)
     elif activation_fun is "ReLu":
         x_current = ReLu(z_curr)
     elif activation_fun is "SOFTMAX":
         x_current = softmax(z_curr)
     return x_current
#function for forward propagation
def forward(X_train,weights,bias,hidden_layers,specification):
    outputs = []
    x_curr = X_train
    for i in range(hidden_layers+1):
        x_prev = x_curr
        w_curr = weights[hidden_layers-i]
        if (specification==1):
            if (i==hidden_layers):
                act_fun = "SOFTMAX"
            else:
                act_fun = "sigmoid"
        elif (specification==2):
            if (i==hidden_layers):
                act_fun = "SOFTMAX"
            else:
                act_fun = "ReLu"
        bias_curr = bias[hidden_layers-i]
        x_curr = single_layer_forward(x_prev,bias_curr,w_curr,act_fun)
        outputs.append(x_curr)
    return outputs

def single_layer_backward(weight_curr,output_prev,delta_curr,act_fun):
    part1 = np.zeros([output_prev.shape[0],output_prev.shape[1]])
    part2 = np.dot(weight_curr,delta_curr)
    if act_fun is "sigmoid":
        for i in range(output_prev.shape[0]):
            for j in range(output_prev.shape[1]):
                part1[i][j] = output_prev[i][j]*(1-output_prev[i][j])
    elif act_fun is "ReLu":
        for k in range(output_prev.shape[0]):
            for l in range(output_prev.shape[1]):
                if (output_prev[k][l] >0):
                    part1[k][l] = 1
                elif (output_prev[k][l] == 0):
                    part1[k][l] = 0
    delta_prev = np.zeros([output_prev.shape[0],output_prev.shape[1]])
    for i in range(output_prev.shape[0]):
        for j in range(output_prev.shape[1]):
            delta_prev[j][i]= (part1[i][j])*(part2[j][i])

    return delta_prev
#function for backward propagation
def backward(x,delta_output,weights,bias,outputs,hidden_layers,spec):
    deltas = []
    updated_weights = []
    updated_bias = []
    learning_rate = 0.01
    no_of_train_exp = x.shape[0]
    delta_curr = delta_output

    for i in range(hidden_layers):
         j = hidden_layers - i - 1
         w_curr = weights[i]
         b_curr = bias[i]
         output_prev = outputs[j]
         dw = np.dot(output_prev.T,delta_curr.T)/no_of_train_exp
         db = np.sum(delta_curr,axis=1)/no_of_train_exp
         updated_wts = w_curr - (learning_rate)*dw
         updated_b = b_curr - (learning_rate)*db
         updated_weights.append(updated_wts)
         updated_bias.append(updated_b)
         if (spec==1):
             act_fun = "sigmoid"
         elif (spec==2):
            act_fun = "ReLu"
         delta_curr = single_layer_backward(w_curr,output_prev,delta_curr,act_fun)
         deltas.append(delta_curr)

    input_dw = np.dot(x.T,delta_curr.T)/ no_of_train_exp  #weights connecting input layer and 1sthid
    input_db = np.sum(delta_curr,axis=1)/no_of_train_exp
    updated_weights.append(input_dw)
    updated_bias.append(input_db)
    return updated_bias,updated_weights
#function to claculate error
def cross_entropy_loss(y,final_output):    #final_output is output_layer values
    m = y.shape[0]
    Q = np.log(final_output)
    E = np.dot(y,Q.T)
    sum = 0
    for i in range(m):
        sum = sum+E[i][i]
    error = -(1/m)*sum
    return error
#function to calculate deltas of output layer
def delta_ouput_layer(y,outputs):
    m = y.shape[0]
    delta_output = np.zeros([3,m])
    fin_output = outputs[-1]
    temp1 = np.zeros([m,3])
    temp2 = np.zeros([m, 3])
    temp3 = np.zeros([m, 3])
    for i in range(3):
        for j in range(m):
            if i == 0:
               temp1[j][i] = y[j][i]*(1-fin_output[j][0])
            else:
               temp1[j][i] = y[j][i]*(-fin_output[j][0])
    for i in range(3):
        for j in range(m):
            if i == 1:
               temp2[j][i] = y[j][i]*(1-fin_output[j][1])
            else:
               temp2[j][i] = y[j][i]*(-fin_output[j][1])
    for i in range(3):
        for j in range(m):
            if i == 2:
               temp3[j][i] = y[j][i]*(1-fin_output[j][2])
            else:
               temp3[j][i] = y[j][i]*(-fin_output[j][2])
    for k in range(m):
       delta_output[0][k]= -(1/m)*np.sum(temp1)
       delta_output[1][k] = -(1/m)*np.sum(temp2)
       delta_output[2][k] = -(1/m)*np.sum(temp3)

    return delta_output
#function to convert probabilities into classes
def con_prob_to_class(outputs):
    final_output = outputs[-1]
    fin_out_arr = np.zeros([len(final_output),3])
    for i in range(len(final_output)):
        fin_out_arr[i] = final_output[i]
    final_class = np.zeros([fin_out_arr.shape[0], fin_out_arr.shape[1]])
    max_index_row = np.argmax(fin_out_arr, axis=1)
    for i in range(fin_out_arr.shape[0]):
        index = max_index_row[i]
        final_class[i][index] = 1
    return final_class
#finction to find accuracy of classifier
def accuracy(outputs,actual_class):
    m = actual_class.shape[0]
    predicted_class = con_prob_to_class(outputs)
    sum = 0
    for i in range(m):
        if np.array_equal(predicted_class[i],actual_class[i]):
            sum = sum+1
    accuracy = (sum/m)*100
    return accuracy

def weight_matrices(specification,hidden_lay):
  Neurons1 = [7,32,3]
  Neurons2 = [7,64,32,3]
  initial_weights = []
  initial_bias = []
  for i in range(hidden_lay+1):
     if specification == 1:
        w = Weight_initialiser(Neurons1[hidden_lay-i],Neurons1[hidden_lay-i+1])
        b = Weight_initialiser(1,Neurons1[hidden_lay-i+1])
        initial_weights.append(w)
        initial_bias.append(b)
     elif specification == 2:
        w = Weight_initialiser(Neurons2[hidden_lay-i], Neurons2[hidden_lay-i+1])
        b = Weight_initialiser(1,Neurons2[hidden_lay-i+1])
        initial_weights.append(w)
        initial_bias.append(b)
  return initial_bias,initial_weights
# function to train Neural network
def train_NN(x_train,y_train,weights,bias,hid_lays,spec,epochs):
    wts = weights
    b = bias
    act_class = one_hot_vect(y_train)
    for i in range(epochs):
       outputs = forward(x_train,wts,b,hid_lays,spec)
       delta_output = delta_ouput_layer(act_class,outputs)
       upd_bias,upd_wts = backward(x_train,delta_output,wts,b,outputs,hid_lays,spec)
       b = upd_bias
       wts = upd_wts
    return wts,b

X_train,y_train,X_test,y_test = preprocess()
Y_train = np.reshape(y_train,(len(y_train),1))
Y_test = np.reshape(y_test,(len(y_test),1))

init_bias1,init_wts1 = weight_matrices(1,1)
init_bias2,init_wts2 = weight_matrices(2,2)
b1 = init_bias1
b2 = init_bias2
w1 = init_wts1
w2 = init_wts2
for i in range(5):
    tuples1 = create_mini_batches(X_train,Y_train,32)[i]
    X = tuples1[0]
    Y = tuples1[1]
    weights,Bias = train_NN(X,Y,w1,b1,1,1,200)
    w1 = weights
    b1 = Bias

print("Part 1 Specification 1A :")
O1_train = forward(X_train,w1,b1,1,1)
actual_class1_train = one_hot_vect(Y_train)
acc1_train = accuracy(O1_train,actual_class1_train)
print("Training accuracy of Task_1A :"+str(acc1_train)+"%")

O1_test =forward(X_test,w1,b1,1,1)
actual_class1_test = one_hot_vect(Y_test)
acc1_test = accuracy(O1_test,actual_class1_test)
print("Test accuracy of Task_1A :"+str(acc1_test)+"%")

print("------------------------------------------------------")
print("Part 1 Specification 1B :")

O2_train = forward(X_train,w2,b2,2,2)
actual_class2_train = one_hot_vect(Y_train)
acc2_train = accuracy(O2_train,actual_class2_train)
print("Training accuracy of Task_1B :"+str(acc2_train)+"%")

O2_test = forward(X_test,w2,b2,2,2)
actual_class2_test = one_hot_vect(Y_test)
acc2_test = accuracy(O2_test,actual_class2_test)
print("Test accuracy of Task_1B :"+str(acc2_test)+"%")



