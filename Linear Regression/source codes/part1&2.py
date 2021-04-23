import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# used pandas just only for reading the csv file
# used numpy only for creating arrays

trainset = pd.read_csv("train.csv")
x = trainset.iloc[:, :-1].values
y = trainset.iloc[:, 1].values

plt.plot(x, y, 'go')
plt.xlabel('features')
plt.ylabel('label')

testset = pd.read_csv("test.csv")
x1 = testset.iloc[:, :-1].values
y1 = testset.iloc[:, 1].values

def hypothesis(theta_arr , x , degree):
    add = 0
    for m in range(degree):
        add = add + (theta_arr[m] * (x ** m))
    return add

deg = input("Enter the degree :")
a = np.ones(10)             # initilizing array of parameters with 1
alpha = 0.05

for iterations in range(5000):
    cost = 0
    temp = np.zeros(10)
    for i in range(trainset.shape[0]):
        cost = cost + (hypothesis(a,x[i],int(deg)+1) - y[i]) ** 2
        for j in range(int(deg)+1):
            if j==0:
               temp[j] = temp[j] + (hypothesis(a,x[i],int(deg)+1) - y[i])
            else:
                temp[j] = temp[j] + (hypothesis(a,x[i],int(deg)+1) - y[i]) * (x[i]**j)
    cost = cost / (2 * trainset.shape[0])
    for l in range(int(deg)+1):
        temp[l] = temp[l] / trainset.shape[0]
    for k in range(int(deg)+1):
        a[k] = a[k] - alpha * temp[k]

print("values of parameters :")
for n in range(int(deg)+1):
    print("a["+str(n)+"]:",a[n])


train_error = 0
y_calculated = []
for i in range(trainset.shape[0]):
    y_calculated.append(hypothesis(a,x[i],int(deg)+1))
    train_error = train_error + (hypothesis(a,x[i],int(deg)+1) - y[i]) ** 2
train_error = train_error / (2 * trainset.shape[0])
plt.plot(x, y_calculated,'bo')
print('Train error =', train_error)

test_error = 0
y_calc_test = []
for i in range(testset.shape[0]):
    y_calc_test.append(hypothesis(a,x1[i],int(deg)+1))
    test_error = test_error + (hypothesis(a,x1[i],int(deg)+1) - y1[i]) ** 2
test_error = test_error / (2 * testset.shape[0])
print('Test error =', test_error)


plt.show()