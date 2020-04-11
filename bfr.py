from pyspark import SparkContext,SparkConf
import os
import itertools
import random
from collections import defaultdict
import math
import sys
import csv
import json
import time

def find_point_belonging(sumer_dict,threshold,d_index,d_features):
        min_distance=float("inf")
        belonging = -1
        for k,v in sumer_dict.items():
            temp_distance = calculate_mahalanobis_distance(d_features,v)
            if temp_distance<min_distance:
                min_distance=temp_distance
                belonging = k
        if min_distance < threshold:
            return belonging,d_index
        else:
            return -1,d_index
        
def calculate_mahalanobis_distance(d_features,summerized):
    centroid=[i/summerized["n"] for i in summerized["SUM"]]
    differences = [x-y for (x,y) in zip(d_features,centroid)]
    variances= [(x/summerized["n"])-(y/summerized["n"])**2 for (x,y) in zip(summerized["SUMSQ"],summerized["SUM"])]
    distance = math.sqrt(sum([(x/math.sqrt(y))**2 for (x,y) in zip(differences,variances)]))
    return distance

def find_nearest(cs_sum_dict,cs_index,cs_indexes,threshold):
    cs1_summerized = cs_sum_dict[cs_index]
    cs_centor = [ x/cs1_summerized["n"] for x in cs1_summerized["SUM"]]
    final_min = float('inf')
    for i in cs_indexes:
        temp_summerized = cs_sum_dict[i]
        temp_dis = calculate_mahalanobis_distance(cs_centor,temp_summerized)
        if temp_dis < final_min:
            final_min = temp_dis
            belonging = i
    if final_min< threshold:
        return cs_index,belonging
    else:
        return cs_index,"no merge"

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

class BFR():
    def __init__(self,inPath,k,outPath1,alpha = 2):
        scConf = SparkConf() \
        .setAppName('hw4') \
        .setMaster('local[1]')
        self.sc = SparkContext(conf=scConf)
        self.sc.setLogLevel("WARN")
        self.root_path = inPath
        self.outPath1 = outPath1
        self.ds_summerized_dict = defaultdict(dict)
        self.first_round = True
        self.alpha = alpha
        self.k = k
        self.ds = defaultdict(list)
        self.cs_summerized_dict =defaultdict(dict)
        self.cs = defaultdict(list)
        self.rs=set()
        self.cs_index=0
        self.intermediate = []
        
    def bfr(self):

        for root, dirs, files in os.walk(self.root_path):
            
            
            for cur_round,file in enumerate(files):
                data_path = os.path.join(self.root_path,file)
                if self.first_round:
                    
                    print("Starting First Round......")
                    data_rdd = self.sc.textFile(data_path).map(lambda x: x.split(",")).map(lambda x:[float(i) for i in x]).map(lambda x: (int(x[0]),x[1:])).cache()
                    self.threshold = self.alpha*(len(data_rdd.take(1)[0][1])**0.5)
                    self.data_dict=dict(data_rdd.collect())
                    
                    sample_data = dict(random.sample(self.data_dict.items(),int(0.1*len(self.data_dict))))
                    
                    temp=KMeans(sample_data,self.k)
                    temp.fit()
                    data_to_be_summerized = temp.clusters_point
                    self.centroids= temp.centroids
                    for k,v in data_to_be_summerized.items():
                        ds_sum,ds_sumsq,ds_n = self.summerize(v)
                        self.ds_summerized_dict[k]["SUM"] = ds_sum
                        self.ds_summerized_dict[k]["SUMSQ"] = ds_sumsq
                        self.ds_summerized_dict[k]["n"] = ds_n  
                        self.ds[k].extend(v)
                    
                    for i in sample_data.keys():
                        self.data_dict.pop(i)
                    cent = self.centroids
                    summerized_dict = self.ds_summerized_dict
                    threshold=self.threshold
                    ds_candidate = self.sc.parallelize(self.data_dict.items()).map(lambda x: find_point_belonging(summerized_dict,threshold,x[0],x[1])).groupByKey().filter(lambda x: x[0] != -1).collectAsMap()
                    
                    for k,v in ds_candidate.items():
                        self.ds[k].extend(list(v))
                        temp_sum,temp_sumsq,temp_n = self.summerize(v)
                        self.update_summerize(k,temp_sum,temp_sumsq,temp_n,True)
                        list(map(self.data_dict.pop, v))
                    
                    print(len(self.data_dict))
                    kmean_to_rest = KMeans(self.data_dict,self.k)
                    kmean_to_rest.fit()
                    
                    for k,v in kmean_to_rest.clusters_point.items():
                        if len(v) == 1:
                            self.rs.add(v[0])
                        else:
                            temp_sum,temp_sumsq,temp_n = self.summerize(v)
                            temp_key= "cs_%d"%self.cs_index
                            self.cs[temp_key].extend(v)
                            self.cs_summerized_dict[temp_key]={
                                "SUM": temp_sum,
                                "SUMSQ": temp_sumsq,
                                "n": temp_n
                            }
                            self.cs_index+=1
                            list(map(self.data_dict.pop, v))
                    
                    self.first_round = False
                    nof_point_discard=sum([len(v) for k,v in self.ds.items()])
                    nof_cluster_compression = len(self.cs_summerized_dict)
                    nof_point_compression = sum([len(v) for k,v in self.cs.items()])
                    nof_point_retained = len(self.rs)
                    self.intermediate.append([cur_round+1, self.k, nof_point_discard, nof_cluster_compression, nof_point_compression,nof_point_retained])
                    print(f"First round finished:\nNumber of CS:\t{nof_cluster_compression}\nNumber of RS:\t{len(self.rs)}\nNumber of Clustered:\t{nof_point_discard}\nRemaining Data:\t{len(self.data_dict)}")
                    
                else:
                    print("Loading new")
                    ds_summerized_dict = self.ds_summerized_dict
                    threshold = self.threshold
                    data_rdd = self.sc.textFile(data_path).map(lambda x: x.split(",")).map(lambda x:[float(i) for i in x]).map(lambda x: (int(x[0]),x[1:]))
                    self.data_dict.update(dict(data_rdd.collect()))
                    print(len(self.data_dict))
                    
                    ds_candidate = data_rdd.map(lambda x: find_point_belonging(ds_summerized_dict,threshold,x[0],x[1])).groupByKey().collectAsMap()

                    points_need_to_be_assigned=[]
                    for k,v in ds_candidate.items():
                        if k != -1:
                            self.ds[k].extend(list(v))
                            temp_sum,temp_sumsq,temp_n = self.summerize(v)
                            self.update_summerize(k,temp_sum,temp_sumsq,temp_n,True)
                            list(map(self.data_dict.pop, v))
                        else:
                            points_need_to_be_assigned.extend(list(v))
                    print(f"Finished generating ds candidate\t|\tRemaining Data:\t{len(self.data_dict)}")
                    cs_summerized_dict = self.cs_summerized_dict
                    points_need_to_be_assigned=list(map(lambda x:(x,self.data_dict[x]),points_need_to_be_assigned))
                    cs_candidate = self.sc.parallelize(points_need_to_be_assigned).map(lambda x: find_point_belonging(cs_summerized_dict,threshold,x[0],x[1])).groupByKey().filter(lambda x: x[0] != -1).collectAsMap()
                    for k,v in cs_candidate.items():
                        temp_sum,temp_sumsq,temp_n = self.summerize(v)
                        temp_key= "cs_%d"%self.cs_index
                        self.cs[temp_key].extend(v)
                        self.cs_summerized_dict[temp_key]={
                                "SUM": temp_sum,
                                "SUMSQ": temp_sumsq,
                                "n": temp_n
                            }
                        self.cs_index += 1
                        list(map(self.data_dict.pop, v))
                    print(f"Finished generating cs candidate\t|\tRemaining Data:\t{len(self.data_dict)}")
                    
                    print("Running K-Means to rest of data")
                    kmean_to_rest = KMeans(self.data_dict,self.k)
                    kmean_to_rest.fit()
                    for k,v in kmean_to_rest.clusters_point.items():
                        if len(v) == 1:
                            self.rs.add(v[0])
                        else:
                            temp_sum,temp_sumsq,temp_n = self.summerize(v)
                            temp_key= "ds_%d"%self.cs_index
                            self.cs[temp_key].extend(v)
                            self.cs_summerized_dict[temp_key]={
                                "SUM": temp_sum,
                                "SUMSQ": temp_sumsq,
                                "n": temp_n
                            }
                            self.cs_index += 1
                            list(map(self.data_dict.pop, v))
                            self.rs = self.rs.difference(set(v))
                    self.merge_cs()
                    nof_point_discard=sum([len(v) for k,v in self.ds.items()])
                    nof_cluster_compression = len(self.cs_summerized_dict)
                    nof_point_compression = sum([len(v) for k,v in self.cs.items()])
                    nof_point_retained = len(self.rs)
                    if cur_round < len(files)-1:
                        self.intermediate.append([cur_round+1, self.k, nof_point_discard, nof_cluster_compression, nof_point_compression,nof_point_retained])
                    print(f"Round finished:\nNumber of CS:\t{len(self.cs_summerized_dict)}\nNumber of RS:\t{len(self.rs)}\nNumber of Clustered:\t{sum([len(v) for k,v in self.ds.items()])}\nRemaining Data:\t{len(self.data_dict)}")

        for k,v in self.cs_summerized_dict.items():
            cs_centroid= [x/v["n"] for x in v["SUM"]]
            ds_belonging = self.find_nearest_ds(cs_centroid)
            self.ds[ds_belonging].extend(self.cs[k])
            self.cs.pop(k)
        self.cs_summerized_dict.clear()
        for i in self.rs:
            cur_point = self.data_dict[i]
            ds_belonging = self.find_nearest_ds(cur_point)
            self.ds[ds_belonging].append(i)
            self.data_dict.pop(i)
        self.rs.clear()
        nof_point_discard=sum([len(v) for k,v in self.ds.items()])
        nof_cluster_compression = len(self.cs_summerized_dict)
        nof_point_compression = sum([len(v) for k,v in self.cs.items()])
        nof_point_retained = len(self.rs)
        self.intermediate.append([cur_round+1, self.k, nof_point_discard, nof_cluster_compression, nof_point_compression,nof_point_retained])
        ds=self.ds
        with open(self.outPath1,"w") as f:
            result = self.sc.parallelize(ds.items()).flatMap(lambda x: [(j,x[0]) for j in x[1]]).sortBy(lambda x: x[0]).collectAsMap()
            json.dump(result,f)
        self.sc.stop()

    def find_nearest_ds(self,center):
        final_min=float("inf")
        for k,v in self.ds_summerized_dict.items():
            temp_dis = calculate_mahalanobis_distance(center,v)
            if temp_dis<final_min:
                belonging = k
                final_min = temp_dis
        return belonging
            
    def merge_cs(self):
        combinations = list(itertools.combinations(sorted(self.cs_summerized_dict.keys()),2))
        cs_summerized_dict = self.cs_summerized_dict
        threshold= self.threshold
        pair_to_be_merged = self.sc.parallelize(combinations).groupByKey().map(lambda x:find_nearest(cs_summerized_dict,x[0],list(x[1]),threshold)).filter(lambda x: x[1] != "no merge").map(lambda x: tuple(sorted(x))).distinct().collect()
        print("Merging CS......")
        for i in pair_to_be_merged:
            cur_summerized = self.cs_summerized_dict[i[0]]
            cur_sum = cur_summerized["SUM"]
            cur_sumsq = cur_summerized["SUMSQ"]
            cur_n = cur_summerized["n"]
            self.update_summerize(i[1],cur_sum,cur_sumsq, cur_n,False)
            self.cs_summerized_dict.pop(i[0])
            self.cs[i[1]].extend(self.cs[i[0]])
            self.cs.pop(i[0])
        
            
    
    
        
        
    def update_summerize(self,cluster_index,addition_sum,addition_sumsq,addition_n,ds_or_not):
        if ds_or_not:
            self.ds_summerized_dict[cluster_index]["SUM"]=[sum(i) for i in zip(self.ds_summerized_dict[cluster_index]["SUM"],addition_sum)]
            self.ds_summerized_dict[cluster_index]["SUMSQ"]=[sum(i) for i in zip(self.ds_summerized_dict[cluster_index]["SUMSQ"],addition_sumsq)]
            self.ds_summerized_dict[cluster_index]["n"] += addition_n
        else:
            self.cs_summerized_dict[cluster_index]["SUM"]=[sum(i) for i in zip(self.cs_summerized_dict[cluster_index].get("SUM",[0]*len(addition_sum)),addition_sum)]
            self.cs_summerized_dict[cluster_index]["SUMSQ"]=[sum(i) for i in zip(self.cs_summerized_dict[cluster_index].get("SUMSQ",[0]*len(addition_sumsq)),addition_sumsq)]
            self.cs_summerized_dict[cluster_index]["n"] = self.cs_summerized_dict[cluster_index].get("n",0) + addition_n
        
    
    def summerize(self,points_index):
        index_to_features = [self.data_dict[i] for i in points_index]
        sum_ = [sum(i) for i in zip(*index_to_features)]
        sumsq = [sum(map(lambda x: x**2,i)) for i in zip(*index_to_features)]
        n = len(points_index)
        return sum_,sumsq,n
        
def main(inFile,n_cluster,outFile_1,outFile_2):
    bfr= BFR(inPath=inFile,k=n_cluster,outPath1=outFile_1)
    bfr.bfr()
    out_f2 = open(outFile_2,"w")
    writer = csv.writer(out_f2)
    writer.writerow(["round_id","nof_cluster_discard","nof_point_discard","nof_cluster_compression","nof_point_compression","nof_point_retained"])
    writer.writerows(bfr.intermediate)
if __name__ =="__main__":
    inPath = sys.argv[1]
    ncluster = int(sys.argv[2])
    outPath_1 = sys.argv[3]
    outPath_2 = sys.argv[4]
    starttime=time.time()
    main(inPath,ncluster,outPath_1,outPath_2)
    print("Total Running Time:\t",time.time()-starttime)






                



    
        







