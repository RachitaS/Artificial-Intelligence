import numpy
import matplotlib.pyplot as plt
import csv
from random import shuffle
import math
import operator 

DataSets = ['iris.data', 'wine.data','winequality-red.csv']
cls = [4,0,-1]
def plotIris():
	with open('iris.data', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		Xpoints=[]
		Ypoints=[]
		colorList=[]
		

		classes={}
		l=0
		nds = []
		for ds in dataset:
			nds.append([ds[3],ds[1],ds[-1]])
			if(ds[-1] not in classes):
				classes[ds[-1]]=l
				l += 1
			Xpoints.append(ds[3])
			Ypoints.append(ds[1])
			colorList.append(classes[ds[-1]])
		plt.axis([0, 3, 0, 4.5])
		a = numpy.array([Xpoints,Ypoints])
		categories = numpy.array(colorList)
		colormap = numpy.array(['r', 'g', 'b'])
		plt.scatter(a[0], a[1], s=50, c=colormap[categories])
		plt.xlabel('Petal Width')
		plt.ylabel('Sepal width')
		xArr = [.2]
		yArr = [2]
		x = .2
		while x < 3:
			y = 4.4
			while y > 2:
				if (getNearest(nds,[x+.1,y]) != getNearest(nds,[x-.1,y])):
					xArr.append(x)
					yArr.append(y)
					break
				y -= .1
			x += .1		
		
		x = 1.5
		while x < 2:
			y = 2
			while y < 4.4:
				if (getNearest(nds,[x+.1,y]) != getNearest(nds,[x-.1,y])):
					xArr.append(x)
					yArr.append(y)
					break
				y += .1
			x += .1		
		
		ylen = len(yArr)-1
		for w in range(ylen):
			plt.plot([xArr[w],xArr[w+1]],[yArr[w],yArr[w+1]])
		plt.show()

def getNearest(d=[], p=[]):
	m = 10;
	ms = []
	for s in d:
		dist = math.sqrt(pow((float(s[0]) - float(p[0])), 2) + pow((float(s[1]) - float(p[1])), 2))
		if dist < m:
			m = dist
			ms = s[-1]
	return ms
	
def distance(instance1, instance2, length):
	distance=0
	for y in range(length):
		if y is not cls[itr]:
			distance += pow((float(instance1[y]) - float(instance2[y])), 2)
	return math.sqrt(distance)

def loadDataset(filename, trainingSet=[] , testSet=[], classes={}):
	with open(filename, 'r') as csvfile:
		if(itr==2):
			lines = csv.reader(csvfile, delimiter=';')
		else:
			lines = csv.reader(csvfile)
		
		dataset = list(lines)
		length=len(dataset)
		if (itr == 2):
			dataset.pop(0)
			length = length-1
			
		#print (length)
		shuffle(dataset)
		l=0
		lb=0
		if len(classes) is 0:
			lb=1

		for i in range (length):
			if(lb is 1 and dataset[i][cls[itr]] not in classes):
				classes[dataset[i][cls[itr]]]=l
				l += 1

			if(i%2==0):
				testSet.append(dataset[i])
			else:
				trainingSet.append(dataset[i])


def loadDataSetsFiveFold(filename, trainingSets=[[],[],[],[],[]] , testSets=[[],[],[],[],[]]):
	with open(filename, 'r') as csvfile:
		if(itr==2):
			lines = csv.reader(csvfile, delimiter=';')
		else:
			lines = csv.reader(csvfile)
		dataset = list(lines)
		length=len(dataset)
		if(itr==2):
			dataset.pop(0)
			length = length -1
		shuffle(dataset)
		foldSize=(length/5)
		for i in range (0,5):
			for y in range (0,length):
				if ((y < (i+1)*foldSize) and (y >= i*foldSize)):
					testSets[i].append(dataset[y])
				else:
					trainingSets[i].append(dataset[y])


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)
	for x in range(0,len(trainingSet)):
		dist = distance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	k=int(k)
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	global itr
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][cls[itr]]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	global itr
	correct = 0
	for x in range(len(testSet)):
		if str(testSet[x][cls[itr]])==str(predictions[x]):
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	global itr
	itr=-1
	print ("Please choose the dataset:")	
	for d in DataSets:
		itr += 1
		print ("Enter ",itr, " for - ",d)
		
	itr = int(input())
	ds=DataSets[itr]

	print ('-----------------------------')
	print ("Dataset",itr," = ", ds)
	print ('-----------------------------')
	print ()

	for k in 1,3:
		print ("Results for ", k,"NN (Dataset : ",ds,")")
		print ()
		print ("Random Sampling (Dataset : ",ds,", K :",k,"):")
		acList=[]
		classes={}
		for i in range (1,11):
			trainingSet=[]
			testSet=[]
			loadDataset(ds, trainingSet, testSet, classes)
			nclass=len(classes.keys())
			predictions=[]
			confusionM=[[0 for i in range(nclass)] for j in range(nclass)]
			for x in range(len(testSet)):
				neighbors = getNeighbors(trainingSet, testSet[x], k)
				result = getResponse(neighbors)
				predictions.append(result)
				confusionM[classes[testSet[x][cls[itr]]]][classes[result]] += 1
			
			accuracy = getAccuracy(testSet, predictions)
			acList.append(accuracy)
			print("Iter ", i ," Accuracy: "+str(accuracy))
			print ("Confusion Matrix",i," - (Dataset : ",ds,", K :",k,"):")
			print (classes)
			
			for cf in confusionM:
				print (cf)
			print (" ")
		
		print("Grand Mean Accuracy: "+str(numpy.mean(acList)))
		print ("Standard Deviation of Accuracy: "+str(numpy.sqrt(numpy.var(acList))))
		

		print ()
		print ("5 cross validation (Dataset : ",ds,", K :",k,"):")
		print ()
		acList=[]
		trainingSets=[[],[],[],[],[]]
		testSets=[[],[],[],[],[]]
		acList=[]
		loadDataSetsFiveFold(ds,trainingSets,testSets)

		for i in range (0,5):
			predictions=[]
			for x in range(len(testSets[i])):
				neighbors = getNeighbors(trainingSets[i], testSets[i][x], k)
				result = getResponse(neighbors)
				predictions.append(result)
			accuracy = getAccuracy(testSets[i], predictions)
			acList.append(accuracy)	
			print(i+1, " fold Accuracy: "+str(numpy.mean(acList)))
		print ()
		print("5-fold Mean: "+str(numpy.mean(acList)))
		print ("5-fold SD: "+str(numpy.sqrt(numpy.var(acList))))
	print ("#############################################################################################")
	print ("#############################################################################################")
	print ()
	if (itr is 0):
		print ("Plotting Iris dataset and decision boundary")
		plotIris()

main()
