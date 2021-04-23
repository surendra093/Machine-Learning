import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfTransformer

dataset = pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")
att = dataset.iloc[:,1:].values
columns = att.shape[1]
rows = att.shape[0]

array = np.zeros([rows,columns], dtype=float)
array = att

transformer = TfidfTransformer(smooth_idf=True)
tfidf = transformer.fit_transform(array)
Tf_Idf = tfidf.toarray()
cosine_sim = np.zeros([rows,rows],dtype=float)
dist_mat = np.zeros([rows,rows],dtype=float)

for i in range(rows):
    for j in range(i+1,rows):
         cosine_sim[i][j] = np.dot(Tf_Idf[i],Tf_Idf[j])
         cosine_sim[j][i] = cosine_sim[i][j]

for k in range(rows):
    for l in range(k+1,rows):
        dist_mat[k][l] = 1/(math.exp(cosine_sim[k][l]))
        dist_mat[l][k] = dist_mat[k][l]

#print(dist_mat)
def similarity():
     simlarity = cosine_sim
     return  simlarity

def distance():
    distance = dist_mat
    return distance

def TF_IDF():
    vectors = Tf_Idf
    return vectors

def main():
    print("TF-IDF scores matrix after normalization:")
    print(Tf_Idf)

if __name__ == '__main__':
     main()
