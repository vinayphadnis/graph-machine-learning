import matplotlib

import math
import json
import numpy as np
import random

matplotlib.use('WebAgg')
plt = matplotlib.pyplot

fig, ax = plt.subplots()

def generate_one_hot_encodings(data, maxData):
    array = []
    for obj in data:
        addressArray = []
        for dataObj in maxData:
            if dataObj in obj: addressArray.append(1)
            else: addressArray.append(0)
        array.append(addressArray)
    return array
def plotGraph(X, Y, maxData):

    ax.scatter(X, Y)
    for i, txt in enumerate(maxData):
        ax.annotate(txt, (X[i], Y[i]))
    plt.show()
def initGraph(maxData):
    randX = []
    randY = []
    for i in range(len(maxData)):
        randX.append(random.uniform(0, 100))
        randY.append(random.uniform(0, 100))
    X = np.array(randX)
    Y = np.array(randY)
    return np.array([X, Y])

def generateIndices(points):
    indices = []
    for i in range(len(points)):
        if (points[i] == 1): indices.append(i)
    return indices

def generateCentroid(graph, points):
    indices = generateIndices(points)

    X = graph[0]
    Y = graph[1]
    Xsum = 0
    Ysum = 0
    for index in indices:
        Xsum += X[index]
        Ysum += Y[index]
    # print("X sum is: ", Xsum, "   Y sum is: ", Ysum)
    centroid = [Xsum/len(indices), Ysum/len(indices)]
    # print("Original centroid: ", centroid)
    return centroid

def movePoint(pointLoc, slope, distance):
    increment = distance / math.sqrt(1 + math.pow(slope, 2))
    p1 = pointLoc + increment
    p2 = pointLoc - increment
    return [p1, p2]

def translate(graph, points, centroid, learning_rate):   
    X = graph[0]
    Y = graph[1] 
    
    centroidX = centroid[0]
    centroidY = centroid[1]   
    indices = generateIndices(points)
    if(len(indices)==1):
        return graph
    for index in indices:
        x_coordinate = X[index]
        y_coordinate = Y[index]
        slope = (centroidY - y_coordinate)/(centroidX - x_coordinate)
        distance = math.sqrt((math.pow((centroidY - y_coordinate), 2))+(math.pow((centroidX - x_coordinate), 2)))
        
        scaledDistance = distance * learning_rate

        

        translatedX = movePoint(x_coordinate, slope, scaledDistance)
        translatedY = movePoint(y_coordinate, slope, scaledDistance)

        # d1 = math.sqrt((math.pow((centroidY - translatedY[0]), 2))+(math.pow((centroidX - translatedX[0]), 2)))
        # print("Distance between centroid and p1: ", d1)
        # d2 = math.sqrt((math.pow((centroidY - translatedY[1]), 2))+(math.pow((centroidX - translatedX[1]), 2)))
        # print("Distance between centroid and p2: ", d2)

        # if (d1 < d2):
        #     newX = translatedX[0]
        #     newY = translatedY[0]
        #     ax.scatter(newX, newY)
        # else:
        #     newX = translatedX[1]
        #     newY = translatedY[1]
        #     ax.scatter(newX, newY)

        if((x_coordinate < translatedX[0] and translatedX[0] < centroidX) or (x_coordinate > translatedX[0] and translatedX[0] > centroidX)):
            X[index] = translatedX[0]
        else: X[index] = translatedX[1]

        if((y_coordinate < translatedY[0] and translatedY[0] < centroidY) or (x_coordinate > translatedY[0] and translatedY[0] > centroidY)):
            Y[index] = translatedY[0]
        else: Y[index] = translatedY[1]                        
    return [X, Y]

def train(graph, dataSet, learning_rate = 0.8):
    untrainedGraph = graph
    for data in dataSet:
        centroid = generateCentroid(untrainedGraph, data)
        untrainedGraph = translate(untrainedGraph, data, centroid, learning_rate)
    return graph

def predict(graph, maxData, predict):
    addressArray = []
    for dataObj in maxData:
        if dataObj in predict: addressArray.append(1)
        else: addressArray.append(0)
    centroid = generateCentroid(graph, addressArray)
    centroidX = centroid[0]
    centroidY = centroid[1]

    distArray = []
    graphX = graph[0]
    graphY = graph[1]
    for i in range(len(graphX)):
        dist = math.sqrt((math.pow((centroidY - graphY[i]), 2))+(math.pow((centroidX - graphX[i]), 2)))
        distArray.append(dist)
    print("DistArray", distArray)
    npMax = np.array(maxData)
    distArray = np.array(distArray)
    inds = distArray.argsort()
    sortedPredictions = npMax[inds]
    predictions = [i for i in sortedPredictions.tolist() if i not in predict] 
    print("Predictions Array", predictions)
    


def write_graph_to_file(graph, allData, filename):
    toWrite = {
        'data' : allData,
        'graph': {
            'X': graph[0].tolist(),
            'Y': graph[1].tolist()
        }
    }
    json_object = json.dumps(toWrite, indent = 4)
    with open(filename, "w") as outfile: 
        outfile.write(json_object) 

def load_from_file(filename):
    with open(filename) as f:
        fileJson = json.load(f)
    maxData = fileJson['data']
    graph = np.array([fileJson['graph']['X'], fileJson['graph']['Y']])
    return maxData, graph




maxData = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
data = [['d'],['a', 'c'],['a', 'c'],['d', 'b', 'f'],['h', 'e'],['i'],['k', 'l', 'b', 'e', 'a'],['e', 'f', 'g'],['g', 'j', 'b'],['h']]
# data = [['a', 'c']]

maxData, graph = load_from_file('graph.json')
print(graph)
# plotGraph(graph[0], graph[1], maxData)

oneHot = generate_one_hot_encodings(data, maxData)

# graph = initGraph(maxData)
trainedGraph = train(graph, oneHot)


predict(trainedGraph, maxData, ['b', 'a'])
# write_graph_to_file(trainedGraph, maxData, 'newgraph.json')
print(graph)

# plotGraph(trainedGraph[0], trainedGraph[1], maxData)