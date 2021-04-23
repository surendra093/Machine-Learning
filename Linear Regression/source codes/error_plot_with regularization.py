import matplotlib.pyplot as plt

lamda= [0.25,0.5,0.75,1.0]
'''
Ridge regression for degree 1(maximum training error)
(Training and test errors for different lambdas)
'''

train_error1 =[0.09968031,0.09968392,0.09968988,0.09969816]
test_error1  =[0.09553497,0.09550858,0.09548461,0.09546303]

'''
Lasso Regression for degree 1(maximum training error)
(Training and test errors for different lambdas)
'''

train_error2 =[0.09967931,0.09967994,0.099681,0.09968248]
test_error2 =[0.09555044,0.09553745,0.09552489,0.09551274]

plt.figure()
plt.plot(lamda,train_error1, color='green')
plt.plot(lamda,test_error1, color='red')
plt.xlabel('lamda')
plt.ylabel('error')
plt.title('Ridge degree 1')

plt.figure()
plt.plot(lamda,train_error2, color='blue')
plt.plot(lamda,test_error2, color='yellow')
plt.xlabel('lamda')
plt.ylabel('error')
plt.title('Lasso degree 1')

plt.show()