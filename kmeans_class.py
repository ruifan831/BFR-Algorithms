import random
from collections import defaultdict

class KMeans():
    def __init__(self,data,k):
        self.data=data
        self.num_cluster = k
        self.clusters_point=defaultdict(list)
        self.centroids=self.generate_init_centroids()
        

    
    def euclidean_distance(self,features_1,features_2):
        eudcli_dis = sum([ (x-y)**2 for (x,y) in zip(features_1,features_2)])**0.5
        return eudcli_dis
    
    def generate_init_centroids(self):
        centroids = defaultdict(list)
        initial_points_index = random.sample(self.data.keys(),min(len(self.data),self.num_cluster))
        for i,index in enumerate(initial_points_index): 
            centroids[i].extend(self.data[index])
        
        return centroids
    
    def calculate_centroid(self,indexs):
        result=[sum(i)/len(i) for i in zip(*[self.data[x] for x in indexs])]
        return result
    
    def check_convergence(self,new_labels,old_labels):
        return all([set(new_labels[i])==set(old_labels[i]) for i in range(min(len(self.data),self.num_cluster))])
    
    def sse_of_clusters(self,centroids,clusters_point):
        sse=0
        for k,v in centroids.items():
            for j in clusters_point[k]:
                sse += self.euclidean_distance(v,self.data[j])**2
        return sse
    def fit(self):
        global_sse=float("inf")
        for _ in range(1):
            temp_centroids,temp_clusters_point = self.kmean_result()
            temp_sse = self.sse_of_clusters(temp_centroids,temp_clusters_point)
            if temp_sse<global_sse:
                global_sse=temp_sse
                self.centroids=temp_centroids
                self.clusters_point=temp_clusters_point
    
    def kmean_result(self):
        # init_centroids = self.generate_init_centroids()
        converged = False
        final_centroids=self.generate_init_centroids()
        final_clusters_point=defaultdict(list)
        while not converged:
            temp_cluster=defaultdict(list)
            for k,v in self.data.items():
                min_distance=float("inf")
                for i in range(min(len(self.data),self.num_cluster)):
                    cur_centroid=final_centroids[i]
                    cur_distance = self.euclidean_distance(v,cur_centroid)
                    if cur_distance<min_distance:
                        min_distance = cur_distance
                        final_label = i
                temp_cluster[final_label].append(k)

            converged=self.check_convergence(temp_cluster,final_clusters_point)
            new_centroid=defaultdict(list)
            for k,v in temp_cluster.items():
                new_centroid[k]=self.calculate_centroid(v)
            final_centroids = new_centroid
            final_clusters_point = temp_cluster
            
        return final_centroids,final_clusters_point