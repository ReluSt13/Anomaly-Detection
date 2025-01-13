import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# 1.1
graph = nx.Graph()
with open("ca-AstroPh.txt", "r") as f:
    for i, line in enumerate(f):
        if i < 4 or i > 1505:
            continue
        nodes = line.split()
        fromNode, toNode = nodes[:2]
        if graph.has_edge(fromNode, toNode):
            graph[fromNode][toNode]['weight'] += 1
        else:
            graph.add_edge(fromNode, toNode, weight=1)
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())

# 1.2
NrOfNeighbors, NrOfEdges, TotalWeightOfEgoNet, PrincipalEigenValue = {}, {}, {}, {}
for node in graph.nodes:
    nrOfNeighbors = len(list(graph.neighbors(node)))
    egoNet = nx.ego_graph(graph, node)
    nrOfEdges = egoNet.number_of_edges()
    totalWeightOfEgoNet = 0
    for edge in egoNet.edges:
        totalWeightOfEgoNet += egoNet[edge[0]][edge[1]]['weight']
    adjacencyMatrix = nx.adjacency_matrix(egoNet)
    eigenValues, eigenVectors = np.linalg.eigh(adjacencyMatrix.toarray())
    principalEigenValue = max(eigenValues)
    NrOfNeighbors[node] = nrOfNeighbors
    NrOfEdges[node] = nrOfEdges
    TotalWeightOfEgoNet[node] = totalWeightOfEgoNet
    PrincipalEigenValue[node] = principalEigenValue

nx.set_node_attributes(graph, NrOfNeighbors, 'NrOfNeighbors')
nx.set_node_attributes(graph, NrOfEdges, 'NrOfEdges')
nx.set_node_attributes(graph, TotalWeightOfEgoNet, 'TotalWeightOfEgoNet')
nx.set_node_attributes(graph, PrincipalEigenValue, 'PrincipalEigenValue')

# 1.3
logE = np.log([graph.nodes[node]['NrOfEdges'] for node in graph.nodes])
logN = np.log([graph.nodes[node]['NrOfNeighbors'] for node in graph.nodes])
lrModel = LinearRegression()
lrModel.fit(logN.reshape(-1, 1), logE)

theta = lrModel.coef_[0]
logC = lrModel.intercept_
C = np.exp(logC)

anomalyScores = {}
for node in graph.nodes:
    realNumberOfEdges = graph.nodes[node]['NrOfEdges']
    realNumberOfNeighbors = graph.nodes[node]['NrOfNeighbors']
    predictedNumberOfEdges = C * realNumberOfNeighbors ** theta
    ratio = max(realNumberOfEdges, predictedNumberOfEdges) / min(realNumberOfEdges, predictedNumberOfEdges)
    anomalyScores[node] = ratio * np.log(np.abs(realNumberOfEdges - predictedNumberOfEdges) + 1)

# 1.4
sortedNodesByAnomalyScores = sorted(anomalyScores.items(), key=lambda x: x[1], reverse=True)
top10Anomalies = [node for node, _ in sortedNodesByAnomalyScores[:10]]
colors = ['red' if node in top10Anomalies else 'blue' for node in graph.nodes]
plt.figure()
pos = nx.spring_layout(graph, seed=42)
nx.draw(graph, pos, node_color=colors, with_labels=False, node_size=10)
plt.savefig("ex1_4.pdf")
plt.show()

# 1.5
normalizedAnomalyScores = {node: (score / np.max(score)) for node, score in anomalyScores.items()}
lof = LocalOutlierFactor()

X = np.array([[graph.nodes[node]['NrOfEdges'], graph.nodes[node]['NrOfNeighbors']] for node in graph.nodes])
lof.fit(X)
lofScores = -lof.negative_outlier_factor_
newAnomalyScores = {node: normalizedAnomalyScores[node] + lofScores[i] for i, node in enumerate(graph.nodes)}

sortedNodesByNewAnomalyScores = sorted(newAnomalyScores.items(), key=lambda x: x[1], reverse=True)
newTop10Anomalies = [node for node, _ in sortedNodesByNewAnomalyScores[:10]]
colors = ['red' if node in newTop10Anomalies else 'blue' for node in graph.nodes]
plt.figure()
nx.draw(graph, pos, node_color=colors, with_labels=False, node_size=10)
plt.savefig("ex1_5.pdf")
plt.show()