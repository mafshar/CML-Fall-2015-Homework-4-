#!usr/bin/env python

'''
Mohammad Afshar
N19624829
CSCI-GA 3033-12: Computational Machine Learning
HW 1
Due: 11/15/15 at 00:00
'''

from sklearn.datasets import load_iris ## obtaining the iris dataset
from scipy.spatial import distance ## for calculating euclidean distance
from sklearn import preprocessing ## for normalizing data
import matplotlib.pyplot as plt
import numpy as np
import random ## for seeding the permutation of the iris data
import copy
import sys

DUE_DATE = 11132015 ## number to seed into the random number gen
COLORS = ["darkred", "darkgreen", "darkblue", "yellow", "grey", "pink",\
            "orange", "brown", "blue", "green", "red", "black" ]

## MYKMEANS ALGO

def display_cluster(c, counter):
    ## get first feature
    x1, x2 = [], []
    for point in c[2]:
        x1.append(point[0])
        x2.append(point[1])
    plt.scatter(x1, x2, c=COLORS[counter])
    plt.scatter(c[1][0], c[1][1], c="orange")
    return

def display_clusters(clusters):
    for i in range(len(clusters)):
        display_cluster(clusters[i], i)
    return

def average_points(points):
    return float(sum(points)) / len(points)

def dist(x,y):
    # print x, y
    return distance.euclidean(x, y)
    # return np.linalg.norm(x-y)
    # return np.sqrt(np.sum((x-y)**2))

def mykmeans(k, data, max_iter):
    if k < 1:
        return
    clusters = []
    random.seed(DUE_DATE)
    for i in range(k):
        centroid = list(random.choice(data)) ## get the centroid
        clusters.append([i, centroid, []])
    for i in range(max_iter):
        for point in data:
            min_distance = sys.float_info.max ## largest float in python
            min_cluster = None
            for c in clusters: ## c == cluster
                ## current distance:
                current = dist(point, c[1])
                if current < min_distance:
                    min_distance = current
                    min_cluster = c
            ## at this point, min_distane and min_cluster must have non-null values
            for c in clusters:
                if c == min_cluster:
                    c[2].append(point)
                    break
        ## now we average the points in each cluster, and make that the centroid
        if i < max_iter - 1:
            for c in clusters:
                upgrade = map(average_points, zip(*c[2])) ## new centroid
                print upgrade
                c[1] = upgrade
                c[2] = []

    display_clusters(clusters)
    return


## MYKMEANS_MULTI ALGO

def randomize_centroids(k, data, centroids):
    np.random.seed(DUE_DATE)
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(\
                            0, len(data), size=1)].flatten().tolist())
    return centroids

def graph_cluster(c, counter, centroids):
    ## get first feature
    x1, x2 = [], []
    for point in c:
        x1.append(point[0])
        x2.append(point[1])
    plt.scatter(x1, x2, c=COLORS[counter])
    for c in centroids:
        plt.scatter(c[0], c[1], c="orange")
    return

def graph_clusters(clusters, centroids):
    for i in range(len(clusters)):
        graph_cluster(clusters[i], i, centroids)
    return

def mykmeans_multi(k, data, max_iter, iterations_to_run):
    np.random.seed(DUE_DATE)
    for i in range (max_iter):
        centroids = []
        previous = [[] for i in range(k)] ## to store previous centroid values
        centroids = randomize_centroids(k, data, centroids)
        for i in range(iterations_to_run):
            clusters = [[] for j in range(k)]
            ## assign point to centroid -- calculating distance (optimized)
            for point in data:
                ## optimization (closest centroid):
                temp_val = min([(i[0], distance.euclidean(point,centroids[i[0]])) \
                                    for i in enumerate(centroids)], \
                                                        key=lambda val:val[1])[0]
                ## small error handling to avoid bad keys
                try:
                    clusters[temp_val].append(point)
                except KeyError:
                    clusters[temp_val] = [point]

            ## test for empty clusters
            for cluster in clusters:
                if not cluster:
                    # optimized
                    cluster.append(\
                            data[np.random.randint(\
                                    0, len(data), size=1)].flatten().tolist())

            ## update centroids (optimized)
            ndx = 0
            for cluster in clusters:
                previous[ndx] = centroids[ndx]
                centroids[ndx] = np.mean(cluster, axis=0).tolist() ## optimized
                ndx += 1

    graph_clusters(clusters, centroids)
    return centroids


## MYKMEANS++ ALGO

def square_dist_prob(data, center):
    temp = []
    for p in data:
        temp.append(distance.euclidean(p, center)**2)
    sumVal = sum(temp)
    probability = []
    for p in data:
        probability.append((distance.euclidean(p, center)**2)/sumVal)
    return probability

def mykmeans_plusplus(k, data, max_iter):
    ## choose a center randomly from the data:
    if k < 1:
        return
    clusters = []
    np.random.seed(DUE_DATE)
    ## get the centroid
    first_ndx = np.random.randint(0, len(data))
    first = copy.deepcopy(data[first_ndx])
    clusters.append([0, first, []])
    index_array = list(range(len(data)))
    for i in range(k-1):
        ndx = np.random.choice(index_array, p=square_dist_prob(data,first))
        centroid = copy.deepcopy(data[ndx])
        clusters.append([i+1, centroid, []])
    for i in range(max_iter):
        for point in data:
            min_distance = sys.float_info.max ## largest float in python
            min_cluster = None
            for c in clusters: ## c == cluster
                ## current distance:
                current = dist(point, c[1])
                if current < min_distance:
                    min_distance = current
                    min_cluster = c
            ## at this point, min_distane and min_cluster must have non-null values
            for c in clusters:
                if c == min_cluster:
                    c[2].append(point)
                    break
        ## now we average the points in each cluster, and make that the centroid
        if i < max_iter - 1:
            for c in clusters:
                upgrade = map(average_points, zip(*c[2])) ## new centroid
                c[1] = upgrade
                c[2] = []

    display_clusters(clusters)
    return


def main():
    plt.xlim(3, 9)
    plt.ylim(1, 6)
    k = 3
    iris = load_iris()
    X, y = iris.data, iris.target
    temp = []
    for i in range(len(X)):
        if y[i] == 1:
            temp.append(X[i])
    data = np.array(temp)
    lissy = copy.deepcopy(data)
    lissy = np.array(temp)
    mykmeans(k, data, 50)
    # plt.clf()
    # mykmeans_multi(k, data, 50, 100)
    # plt.clf()
    # mykmeans_plusplus(k, data, 50)
    plt.show()

    return


    ## REGULAR K-MEANS
    # kmeans(3, data, 50)


if __name__ == "__main__":
    main()
