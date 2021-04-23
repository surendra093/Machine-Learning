import matplotlib.pyplot as plt
import pandas as pd

trainset = pd.read_csv("train.csv")
x = trainset.iloc[:, :-1].values
y = trainset.iloc[:, 1].values

plt.plot(x, y, 'go')
plt.xlabel('features')
plt.ylabel('label')

testset = pd.read_csv("test.csv")
x1 = testset.iloc[:, :-1].values
y1 = testset.iloc[:, 1].values

plt.figure()
plt.plot(x1, y1, 'ro')
plt.xlabel('features')
plt.ylabel('label')

train_error = [0.0996791,0.0995198,0.08861233,0.07189233,0.05707761,0.0465635,0.04045373,0.03773837,0.03718263]
test_error = [0.09556384,0.09548911,0.08657,0.07261383,0.05918206,0.049099,0.04301591,0.0402242,0.03960912]

degree=[1,2,3,4,5,6,7,8,9]

plt.figure()
plt.plot(degree,train_error)
plt.xlabel('degree')
plt.ylabel('training_set error')

plt.figure()
plt.plot(degree,test_error)
plt.xlabel('degree')
plt.ylabel('test_set error')

plt.show()
difference = []
for i in range(9):
    diff = train_error[i] - test_error[i]
    difference.append(diff)
print(difference)