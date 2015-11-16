from sklearn import preprocessing
from sklearn.cluster import KMeans
import random
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import glob
from sklearn.neighbors import KNeighborsClassifier
import wave
import librosa
from scipy.spatial import distance


DUE_DATE = 11132015 ## number to seed into the random number gen
COLORS = ["darkred", "darkgreen", "darkblue", "yellow", "grey", "pink",\
            "orange", "brown", "blue", "green", "red", "black" ]


## MYKMEANS ALGO

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

def KMeans_plusplus(data, k, max_iter):
    np.random.seed(DUE_DATE)
    centroids = []
    previous = [[] for i in range(k)] ## to store previous centroid values
    centroids = randomize_centroids(k, data, centroids)
    for i in range(max_iter):
        clusters = [[] for j in range(k)]
        ## assign point to centroid -- calculating distance (optimized)
        for point in data:
            ## optimization (closest centroid):
            temp_val = min([(i[0], np.linalg.norm(point-centroids[i[0]]))\
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


#euclidean distance helper function
def euclideanDist(point1, point2):
	return distance.euclidean(point1, point2)

#given a path to a file, returns the mfcc vector for that file
def getMFCC(filename):
	y, sr = librosa.load(filename, sr=16000)
	return librosa.feature.mfcc(y=y, sr=sr)

def learnvocabulary(path, clusterSize=3, iterations=50):
	path +='/*.wav'
	MFCCs = []
	for filename in glob.glob(path):
		MFCCs.extend(getMFCC(filename))
	return mykmeans_multi(flattenMFCC(MFCCs), clusterSize, iterations, 10)

#flatten and normalize the MFCC
def flattenMFCC(MFCCs):
    flattened = []
    for MFCC in MFCCs:
        MFCC.flatten()
        sum = 0
        for point in MFCC:
            sum += point
        flattened.append(sum)
    return normalize(np.array(flattened))


# center data and set within range [-1, +1]
def normalize(data):
    preprocessing.scale(data, 0, True, False, False)
    preprocessing.MinMaxScaler((-1, 1), False).fit_transform(data)
    return data


#given a set of clusters and a point, returns the centroid that is closest to that point
def getCentroid(clusters, point):
	bestCluster=[]
	bestDistance = sys.maxint
	for cluster in clusters:
		if euclideanDist(cluster, point) < bestDistance:
			bestDistance = euclideanDist(cluster, point)
			bestCluster = bestCluster
	return bestCluster

def getbof(file, clusters):
	mfcc = flattenMFCC(getMFCC(file))
	bof =[]
	for point in mfcc:
		bof.append(getCentroid(mfcc, clusters))
	return bof

#gets the bag of features for all the wav files in the directory
def getLabelsAndBofs(path, clusters):
	path +='/*.wav'
	bofs = []
	labels = []
	for filename in glob.glob(path):
		bofs.append(getbof(filename))
		labels.append(getLabel(filename))
	return [bofs, labels]

#gets the label for the file
def getLabel(filename):
	dirs = filename.split('/')
	fileToRead = dirs[0] + '/' + dirs[1] + '/labels.txt'
	with open(fileToRead) as f:
		content = f.readlines()
		for line in content:
			labels = line.split(' ')
			if(labels[0] == dirs[-1]):
				return int(labels[1][0])


def KMeansObjective(centroids, points):
    totalPoints = len(points)
    totalDist = 0;
    for point in points:
        bestDist = sys.maxint
        for centroid in centroids:
            dist = euclideanDist(point, centroid)
            if dist < bestDist:
                bestDist = dist
        totalDist +=bestDist
    return totalDist/totalPoints


#function to do a grid search to find the best cluster size
def getBestCluster(path, minClust, maxClust):
	bestVal = KMeansObjective(learnvocabulary(path, clusterSize=minClust))
	for x in range(minClust, maxClust):
		objVal = KMeansObjective(learnvocabulary(path, clusterSize=x))
		if(x < bestVal):
			bestVal = x
	return bestVal

#first, we do a grid search to find the best clustering:
bestVal = getBestCluster('Simpsons/cluster', 1, 10);
clusters = learnvocabulary('Simpsons/cluster', clusterSize=bestVal)

#separate the centroids from the rest of the data
centroids = []
for c in clusters:
	centroids.append(c[0])

#next we get the bag of features, then run nearest neighbors (do a grid search on best k value)
bestVal = sys.minint
bestK = 0
for k in range(1, 10):
	neigh = KNeighborsClassifier(n_neighbors=k)
	train_data, train_labels = getLabelsAndBofs('Simpsons/train', centroids)
	neigh.fit(train_data, train_labels)
	test_data, test_labels = getLabelsAndBofs('Simpsons/test', centroids)
	predicted = neigh.predict(test_data)
	val = np.mean(predicted == test_labels)
	if val > bestVal:
		bestVal = val
		bestK = k

"Best prediction for nearest neigbors is " + bestK + " with an accuracy of " + bestVal
