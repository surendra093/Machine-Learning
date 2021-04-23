import numpy as np
from Task_1 import TF_IDF

Vectors = TF_IDF()

def similarity_matrix(data):
    rows = data.shape[0]
    cosine_sim = np.zeros([rows,rows])
    for i in range(rows):
        for j in range(i + 1, rows):
            cosine_sim[i][j] = np.dot(data[i], data[j])
            cosine_sim[j][i] = cosine_sim[i][j]
    return cosine_sim

def get_clusters(data, cluster_group, n_clusters):
    merged_up_clusters = []
    n_rows_original = cluster_group.shape[0]
    n_rows = cluster_group.shape[0]
    similarity_mat = similarity_matrix(data)
    while (n_rows > 8):
        max_sim = np.amax(similarity_mat)
        max_sim_index = np.where(similarity_mat == max_sim)[0]
        if (cluster_group[max_sim_index[0], -1] == 1):
            ToBeMergedCluster = max_sim_index[0]
            MergingCluster = max_sim_index[1]
        else:
            MergingCluster = int(max_sim_index[0])
            ToBeMergedCluster = int(max_sim_index[1])

        n_g1 = cluster_group[MergingCluster, -1]
        n_g2 = cluster_group[ToBeMergedCluster, -1]

        for i in range(n_g1, n_g1 + n_g2):
            cluster_group[MergingCluster, i] = cluster_group[ToBeMergedCluster, i - n_g1]
        cluster_group[MergingCluster, -1] = n_g1 + n_g2
        for i in range(0, n_rows_original):
            similarity_mat[MergingCluster, i] = max(similarity_mat[MergingCluster, i],
                                                    similarity_mat[ToBeMergedCluster, i])
            similarity_mat[i, MergingCluster] = similarity_mat[MergingCluster, i]
            similarity_mat[MergingCluster, MergingCluster] = 0
        for i in range(0, n_rows_original):
            similarity_mat[ToBeMergedCluster, i] = 0
            similarity_mat[i, ToBeMergedCluster] = 0
        n_rows = n_rows - 1
        merged_up_clusters.append(ToBeMergedCluster)
    cluster_group = np.delete(cluster_group, merged_up_clusters, axis=0)
    return cluster_group

def AgglomerativeClustering(data,n_clusters):
    n_rows=data.shape[0]
    cluster_group=np.zeros((n_rows,n_rows+1),dtype=int)
    for i in range(0,n_rows):
        cluster_group[i,0]=i
        cluster_group[i,-1]=1
    cluster_group_obtained=get_clusters(data,cluster_group,n_clusters)
    cluster_list=[]
    for i in range(0,n_clusters):
        n=cluster_group_obtained[i,-1]
        cluster_list.append(list(cluster_group_obtained[i,:n]))
    return sort(cluster_list)


def sort(clusterlist):
    array = []
    length = len(clusterlist)
    for i in range(0, length):
        array.append(sorted(clusterlist[i]))
    arr1 = []
    for i in range(0,length):
        x = array[i]
        lis = [i, x[0]]
        arr1.append(lis)
    arr1.sort(key=lambda lis: lis[1])

    arr2 = []
    for i in range(0, length):
        arr2.append(array[arr1[i][0]])
    return arr2

def  Final_clusters():
     return AgglomerativeClustering(Vectors,8)

def main():
   print("______________All the eight clusters_______________________")
   for i in range(8):
        print(AgglomerativeClustering(Vectors,8)[i])

if __name__ == '__main__':
     main()