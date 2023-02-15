import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
os.chdir("C:/Users/Trystan/Documents/MachineLearning")

#Question 1

dFProblem6 = [[],[],[]]

D1 = pd.read_csv("data/D1.txt", sep=" ", header=None)
D1.columns = ["X1",'X2','Y']


D2 = pd.read_csv('data/D2.txt', sep=" ", header=None)
D2.columns = ["X1",'X2','Y']


D3leaves = pd.read_csv('data/D3leaves.txt', sep=" ", header=None)
D3leaves.columns = ["X1",'X2','Y']


Dbig = pd.read_csv('data/Dbig.txt', sep=" ", header=None)
Dbig.columns = ["X1",'X2','Y']

Druns = pd.read_csv('data/Druns.txt', sep=" ", header=None)
Druns.columns = ["X1",'X2','Y']

# class Node():

#     def __init__(self, split=None, value=None):
#         self.split = split
#         self.value = value         # for leaf node
#         self.left = Node()
#         self.right = Node()

#     def ApplyDecisionOrLeaf(self, dF):
#         if self.split == None:
#             return self.value
#         dFOutLeft, dFOutRight = ImplementSplit(dF, self.split)

#         self.left.ApplyDecisionOrLeaf(dFOutLeft)

#     # def AddLeftNode(self, childSplit = None, childValue = None):
#     #     self.left = Node(childSplit, childValue)

#     def ApplyNodeOnDataPoint(row, node):
#         if node.split == None:
#             return node.value
#         x = row[node.split[0]]
#         if x >= node.split[1]:
#             return Node.ApplyNodeOnDataPoint(row, node.left)
#         else:
#             return Node.ApplyNodeOnDataPoint(row, node.right)

#     def AddLeftOrRightChild(self, childNode, boolLeft):
#         if boolLeft:
#             self.left = childNode
#         else:
#             self.right = childNode


# def ApplyTree(dF, startingNode):
#     dFOut = pd.DataFrame()
#     for i, row in dF.iterrows():
#         row2 = row
#         row2["Y"] = startingNode.ApplyTreeOnDataPoint(row, startingNode)
#         pd.concat(dFOut, row)
#     return dFOut

def GenerateCandidateSplits(dataFrame):
    X1_Y_Data_List = dataFrame[["X1", "Y"]]
    X2_Y_Data_List = dataFrame[["X2", "Y"]]

    X1_Y_Data_List = X1_Y_Data_List.sort_values(by = "X1").reset_index(drop=True)
    X2_Y_Data_List = X2_Y_Data_List.sort_values(by = "X2").reset_index(drop=True)

    candidateSplits = []

    for d in range(len(X1_Y_Data_List)-1):
        X1_Val = X1_Y_Data_List["X1"][d+1]
        Y_1 = X1_Y_Data_List["Y"][d]
        Y_2 = X1_Y_Data_List["Y"][d+1]
        if Y_1 != Y_2:
            candidateSplits.append(("X1", X1_Val))
            
    for d in range(len(X2_Y_Data_List)-1):
        X2_Val = X2_Y_Data_List["X2"][d+1]
        Y_1 = X2_Y_Data_List["Y"][d]
        Y_2 = X2_Y_Data_List["Y"][d+1]
        if Y_1 != Y_2:
            candidateSplits.append(("X2", X2_Val))

    return candidateSplits

def Entropy(Column):
    x, Unique_class_counts = np.unique(Column, return_counts = True)
    probabilities = Unique_class_counts / Unique_class_counts.sum()
    return EntropyFromProbList(probabilities)

def EntropyFromProbList(probabilities):
    entropy = 0
    for p in probabilities:
        entropy += EntropyTerm(p) #iterate adding Entropy terms
    return entropy

def EntropyTerm(p):
    if p == 0:
        return 0
    return - p * np.log2(p)

def ImplementSplit(dataFrame, split):
    label = split[0]
    split_value = split[1]
    dataFrame = dataFrame.sort_values(by=label).reset_index(drop=True)

    index = dataFrame[dataFrame[label] == split_value].index.astype(int)[0]

    outFrameLeft = dataFrame[index:]
    outFrameRight = dataFrame[:index]
    
    return outFrameLeft, outFrameRight

def ConditionalAndSplitEntropies(dataFrame, split):
    dataFrameLeft, dataFrameRight = ImplementSplit(dataFrame, split)
    splitEntropy = 0 # H_D(S)
    dataCount = len(dataFrame)
    ratioLow = len(dataFrameLeft) / dataCount
    ratioHigh = len(dataFrameRight) / dataCount
    conditionalEntropy = Entropy(dataFrameLeft['Y']) * ratioLow + Entropy(dataFrameRight['Y'])* ratioHigh # H_D(Y|S)
    if ratioLow != 0:
        splitEntropy -= ratioLow * np.log2(ratioLow)
    if ratioHigh != 0:
        splitEntropy -= ratioHigh * np.log2(ratioHigh)
    return conditionalEntropy, splitEntropy

def StoppingCriteria(dataFrame):
    return len(dataFrame) <= 1 or Homogeneity(dataFrame)

def BestSplit(dataFrame):
    if StoppingCriteria(dataFrame):
        return None
    possible_splits = GenerateCandidateSplits(dataFrame)
    if len(possible_splits) == 0:
        return None

    dataEntropy = Entropy(dataFrame["Y"]) # H_D(Y)
    bestGainRatio = 0
    bestSplit = None

    stoppingFlag = True

    for candidate_split in possible_splits:
        conditionalEntropy, splitEntropy = ConditionalAndSplitEntropies(dataFrame, candidate_split) # H_D(Y|S) and H_D(S)
        infoGain = dataEntropy - conditionalEntropy
        if splitEntropy == 0:
            return None
        else:
            gainRatio = infoGain / splitEntropy
        if gainRatio != 0:
            stoppingFlag = False
        # print('label', candidate_feature, 'gainRatio', gainRatio)
        if gainRatio >= bestGainRatio:
            bestGainRatio = gainRatio
            bestSplit = candidate_split

    if stoppingFlag:
        return None

    return bestSplit

def Homogeneity(dataFrame): # Critera 1
    Y = dataFrame["Y"]
    Unique_values = np.unique(Y)
    return len(Unique_values) == 1

def ZeroSplitEntropy(SplitList): #criteria 2
    if any(np.array(SplitList) == 0):
        return True
    else:
        return False

def GenerateLeaf(dataFrame, graph, parentSplit):
    Y = dataFrame["Y"]
    parentNodeName = NodeNameFromSplit(parentSplit)

    leafLabel = "0"
    y = 0
    if np.sum(Y) >= len(Y)/2: #majority
        leafLabel = "1"
        y = 1

    for i, row in dataFrame.iterrows():
        dFProblem6[0].append(row["X1"])
        dFProblem6[1].append(row["X2"])
        dFProblem6[2].append(y)

    leafNodeName = parentNodeName + leafLabel

    graph.node(leafNodeName, leafLabel)
    graph.edge(parentNodeName, leafNodeName)

def NodeNameFromSplit(split):
    if split == None:
        return ""
    return str(split[0]) + " >= " + str(split[1])

def GenerateInternalNode(graph, childSplit, parentSplit):
    childNodeName = NodeNameFromSplit(childSplit)
    graph.node(childNodeName)
    if parentSplit != None:
        parentNodeName = NodeNameFromSplit(parentSplit)
        graph.edge(parentNodeName, childNodeName)
        parentSplit
        

def GenerateSubtree(dataFrame, gViz, parentSplit):
    print("subtree start")
    bestSplit = BestSplit(dataFrame)
    if bestSplit == None:
        print("bestSplit == None")
        GenerateLeaf(dataFrame, gViz, parentSplit)
    else:
        print("bestSplit != None")
        outFrameLeft, outFrameRight = ImplementSplit(dataFrame, bestSplit)
        print("split implemented")
        GenerateInternalNode(gViz, bestSplit, parentSplit)
        print("node generated")
        GenerateSubtree(outFrameLeft, gViz, bestSplit)
        GenerateSubtree(outFrameRight, gViz, bestSplit)
        print("child subtrees done")

def VisualizeTree(dataFileName, FileName):
    gViz = graphviz.Digraph()
    dataFrame = pd.read_csv("data/" + dataFileName, sep=" ", header=None)
    dataFrame.columns = ["X1","X2","Y"]
    dFProblem6 = [[], [], []]
    GenerateSubtree(dataFrame, gViz, None)
    gViz.render(FileName, view=True)
    print("Finished")
    return gViz

# VisualizeTree("D2.txt", "D2TreeTest") ##produces the visual representation of the D2 Tree

def GenerateTree(dataFrame, FileName):
    GenerateSubtree(dataFrame, None, None)
    gViz = graphviz.Digraph()
    parentSplit = 0
    rules = GenerateSubtree(dataFrame, gViz, parentSplit)
    print("Finished")
    return rules

#2.2
Handcraft = pd.DataFrame([[0,1,0],[0,1,1],[2,3,0],[2,3,1]])
Handcraft.columns = ["X1",'X2','Y']
colsMap= Handcraft['Y'].map({0:'g', 1:'r'})
plt.scatter(Handcraft['X1'], Handcraft['X2'],c=colsMap)
plt.show()
#impossible to split data as the data sets completely overlap

# print(Dataset_1)

#2.3
print(GenerateCandidateSplits(Druns))

def GainRatio(dataFrame, candidate_split):
    dataEntropy = Entropy(dataFrame["Y"])
    conditionalEntropy, splitEntropy = ConditionalAndSplitEntropies(dataFrame, candidate_split) # H_D(Y|S) and H_D(S)
    infoGain = dataEntropy - conditionalEntropy
    if splitEntropy == 0:
        return None
    else:
        gainRatio = infoGain / splitEntropy
    return gainRatio

def InformationGain(dataFrame, candidate_split):
    dataEntropy = Entropy(dataFrame["Y"])
    conditionalEntropy, y = ConditionalAndSplitEntropies(dataFrame, candidate_split)
    infoGain = dataEntropy - conditionalEntropy
    return infoGain

def GainListModified(dataFrame, splits):
    gainList = []
    for split in splits:
        if Entropy(dataFrame["Y"]) == 0 or Entropy(dataFrame["Y"]) == None:
            gainList.append(InformationGain(dataFrame, split))
        else:    
            gainList.append(GainRatio(dataFrame,split))
    # take dataFrame, iterate through all splits and add the gainratio to a list
    return gainList

print(GainListModified(Druns, GenerateCandidateSplits(Druns)))




2.6
cols= D1['Y'].map({0:'g', 1:'r'})
print(cols.shape)
plt.scatter(D1['X1'], D1['X2'],c=cols)
plt.show()

cols= D2['Y'].map({0:'g', 1:'r'})
print(cols.shape)
plt.scatter(D2['X1'], D2['X2'],c=cols)
plt.show()

cols = dFProblem6[2]
for i in range(len(cols)):
    if cols[i] == 0:
        cols[i] = 'r'
    else:
        cols[i] = 'g'

print(cols.shape)
plt.scatter(dFProblem6[0], dFProblem6[1], c=cols, linewidths = 3)
plt.show()

#2.7
dBigRand = Dbig.sample(frac=1).iloc[:8129]
dBigRandTest = dBigRand.iloc[8129:]
dBigRandTest.columns = ["X1",'X2','Y']
dBigRand2048 = dBigRand[:2048]
dBigRand2048.columns = ["X1",'X2','Y']
dBigRand512 = dBigRand[:512]
dBigRand512.columns = ["X1",'X2','Y']
dBigRand128 = dBigRand[:128]
dBigRand128.columns = ["X1",'X2','Y']
dBigRand32 = dBigRand[:32]
print(dBigRand32)
dBigRand32.columns = ["X1",'X2','Y']
d2048Rules = GenerateTree(dBigRand2048, "dBig2048")[1]
d512Rules = GenerateTree(dBigRand512, "dBig512")[1]
d128Rules = GenerateTree(dBigRand128, "dBig128")[1]
d32Rules = GenerateTree(dBigRand32, "dBig32")[1]
