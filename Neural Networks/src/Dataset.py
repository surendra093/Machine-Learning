import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#Z_score normalization function to normalize attributes
def z_score(df):
            df.columns = [x + "_zscore" for x in df.columns.tolist()]
            return ((df - df.mean())/df.std(ddof=0))

dataset = pd.read_csv("seed.txt",sep='\t')
att = dataset.iloc[:,:-1].values
dataframe = dataset.iloc[:,:-1]
columns = att.shape[1]
rows = att.shape[0]

norm_array = z_score(dataframe).values
X = norm_array
Y = dataset.iloc[:,-1].values
#split dataset into Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def train_dataset():
    x = X_train
    y = y_train
    return x,y
def test_dataset():
    x = X_test
    y = y_test
    return x,y

def main():
    print('Matrix of Normalized attributes:')
    print(norm_array)

if __name__ == '__main__':
      main()