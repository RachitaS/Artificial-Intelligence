#!/usr/bin/env python
#coding: utf-8

from matplotlib.pylab import *
import random

class Perceptron(object):
	def __init__(self):
		super(Perceptron, self).__init__()
		self.w = [3,2,1] # weights
		self.learningRate = .02
		self.margin = .02
		self.marginVec = [.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05]
		self.theta = 0.001
	
	def testPrediction(self, test, weight):
		c = 0
		i = 0
		for t in test:
			if(weight[0]*t[0] + weight[1]*t[1] + weight[2]*t[2] >= 0):
				print (t,": Actual class= ",t[3], ", Predicted class = ",t[3])
				c += 1
			else:
				print (t,": Actual class= ",t[3], ", Predicted class = ",-t[3])
				i += 1
		accuracy = c *100 / float (c+i)
		print ("Accuracy = ", accuracy)
		

	def response(self, x, op):
		"""perceptron output"""
		y = x[0] * self.w[0] + x[1] * self.w[1] + x[2] * self.w[2]# dot product between w and x
		if(op == 1):
			if y >= 0:
				return 1
			else:
				return -1
		elif (op == 2 or op == 3):
			if y >= self.margin:
				return 1
			else:
				return -1

	def updateWeights(self, x, op, k):
		dt = x[0] * self.w[0] + x[1] * self.w[1] + x[2] * self.w[2]
		if (op==1 or op==2):
			self.w[0] += self.learningRate * x[0]
			self.w[1] += self.learningRate * x[1]
			self.w[2] += self.learningRate * x[2]
		elif (op==3):
			b = self.margin
			m = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
			term = (b - dt)/float(m)
			r=1
			self.w[0] +=  r*term * x[0]
			self.w[1] +=  r*term * x[1]
			self.w[2] +=  r*term * x[2]
		elif (op==4):
			n = len(x)
			r = (self.learningRate/float (k+1)) * (self.marginVec[k] - dt)
			self.w[0] += r * x[0]
			self.w[1] += r * x[1]
			self.w[2] += r * x[2]

	def train(self, data, op):
		k=0
		n = len(data)
		flag = 0
		v = [1,1,1]
		if (op==4):
			self.w = [.05,.2,-.6] # weights
			self.learningRate = .001
		iteration = 0
		while True:
			if (op==4):
				self.updateWeights(data[k], op, k)
				dt = (data[k][0] * self.w[0]) + (data[k][1] * self.w[1]) + (data[k][2] * self.w[2])
				v[0] = data[k][0]
				v[1] = data[k][1]
				v[2] = data[k][2]
				nor = float (v[0] - v[1] + v[2])
				term = float (self.learningRate/(k+1)) * float (self.marginVec[k] - dt)
				term = abs(term * nor)
				if (term <= self.theta):
					flag += 1
				else:
					flag = 0
				if (flag >= n or iteration>100000):
					break
			else:
				r = self.response(data[k], op)
				if (r<=0): # if have a wrong response
					self.updateWeights(data[k], op, 0)
					flag = 0
				else:
					flag += 1
				if (flag > n or iteration>100000):
					break
			iteration += 1
			k = (k+1)%n
		print ("Iteration ",op," = ",iteration)
	
