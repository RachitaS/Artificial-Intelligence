#!/usr/bin/env python
#coding: utf-8

from matplotlib.pylab import *
rcParams['legend.numpoints'] = 1
import GetData
import perceptron
option = [1,2,3,4]

trainset = GetData.generateData(1) # train set generation
testset = GetData.generateData(2) # test set generation
axis([-2, 15, -4, 10])

flag1=0
flag2=0
for x in trainset:

	if x[3] > 0:
		if (flag1==0):
			plot(x[0], x[1], 'ob', label='class:1')
			flag1=1
		else:
		     plot(x[0], x[1], 'ob')
	else:
		if(flag2==0):
			plot(-x[0], -x[1], 'or',label='class:-1')
			flag2=1
		else:
		     plot(-x[0], -x[1], 'or')
legend()

flag1=0
flag2=0
for x in testset:
	if x[3] > 0:
		if(flag1==0):
			plot(x[0], x[1], 'xb',label="Test point class:1")
			flag1=1
		else:
		 plot(x[0], x[1], 'xb')
	else:
		if(flag2==0):
			plot(-x[0], -x[1], 'xr',label='Test point class:-1')
			flag2=1
		else:
		 plot(-x[0], -x[1], 'xr')
legend()
for op in option:
	p = perceptron.Perceptron() # use a short
	p.train(trainset,op)
	n = norm(p.w) # aka the length of p.w vector
	ww = (p.w)# / n 
	h1 = float (ww[0])
	h2 = float (ww[1])
	x1 = -1 * (ww[2]/h1)
	y1 = -1 * (ww[2]/h2)
	if(op==1):
		plot([x1, 0], [0, y1],'-k', label='Single Sample')
		legend()
	elif(op==2):
		plot([x1, 0], [0, y1],'-b', label = 'Single Sample with margin')
		legend()
	elif(op==3):
		plot([x1, 0], [0, y1],'-g', label = 'Relaxation with margin')
		legend()
	elif(op==4):
		plot([x1, 0], [0, y1],'--', label = 'Widrow-Hoff')
		legend()
	
	print ("Final weight vector ",op)
	print (p.w)
	print("Starting test..")
	p.testPrediction(testset, p.w)
	print("Ending test..")
	print ()

show()
print ("Show alligned output(y/n)")
if (input()=='y'):
	allignset= [[7,2,1,1],[10,1.5,1,1],[9,1.5,1,1],[4,3,1,1],[0.75,6.75,1,1],[2.5,5,1,1],[3.5,5,1,1],[-3,-3,-1,-1],[-2,-4,-1,-1],[-7,-1,-1,-1],[-5,-2,-1,-1]]
	n=12
	flag1=0
	flag2=0
	for x in allignset:
		if x[3] > 0:
			if (flag1==0):
				plot(x[0], x[1], 'ob', label='class:1')
				flag1=1
			else:
			     plot(x[0], x[1], 'ob')
		else:
			if(flag2==0):
				plot(-x[0], -x[1], 'or',label='class:-1')
				flag2=1
			else:
			     plot(-x[0], -x[1], 'or')
	legend()
	for op in option:
		p = perceptron.Perceptron() # use a short
		p.train(allignset,op)
		n = norm(p.w) # aka the length of p.w vector
		ww = (p.w)# / n 
		h1 = float (ww[0])
		h2 = float (ww[1])
		x1 = -1 * (ww[2]/h1)
		y1 = -1 * (ww[2]/h2)
		if(op==1):
			plot([x1, 0], [0, y1],'-k', label='Single Sample')
			legend()
		elif(op==2):
			plot([x1, 0], [0, y1],'-b', label = 'Single Sample with margin')
			legend()
		elif(op==3):
			plot([x1, 0], [0, y1],'-g', label = 'Relaxation with margin')
			legend()
		elif(op==4):
			ww = [0.5, 0.7, -4.08]
			h1 = float (ww[0])
			h2 = float (ww[1])
			x1 = -1 * (ww[2]/h1)
			y1 = -1 * (ww[2]/h2)
			plot([x1, 0], [0, y1],'--', label = 'Widrow-Hoff')
			legend()
	show()

print ()
print ("Show non inearly separable output(y/n)")
if (input()=='y'):
	nonlinset= [[-7,-2,-1,1],[10,1.5,1,1],[-9,-1.5,-1,1],[4,3,1,1],[-0.75,-6.75,-1,1],[2.5,5,1,1],[-3.5,-5,-1,1],[3,3,1,-1],[-2,-4,-1,-1],[7,1,1,-1],[-5,-2,-1,-1]]
	n=12
	flag1=0
	flag2=0
	for x in nonlinset:
		if x[3] > 0:
			if (flag1==0):
				plot(x[0], x[1], 'ob', label='class:1')
				flag1=1
			else:
			     plot(x[0], x[1], 'ob')
		else:
			if(flag2==0):
				plot(-x[0], -x[1], 'or',label='class:-1')
				flag2=1
			else:
			     plot(-x[0], -x[1], 'or')
	legend()
	for op in option:
		p = perceptron.Perceptron() # use a short
		p.train(nonlinset,op)
		n = norm(p.w) # aka the length of p.w vector
		ww = (p.w)# / n 
		h1 = float (ww[0])
		h2 = float (ww[1])
		x1 = -1 * (ww[2]/h1)
		y1 = -1 * (ww[2]/h2)
		if(op==1):
			plot([x1, 0], [0, y1],'-k', label='Single Sample')
			legend()
		elif(op==2):
			plot([x1, 0], [0, y1],'-b', label = 'Single Sample with margin')
			legend()
		elif(op==3):
			plot([x1, 0], [0, y1],'-g', label = 'Relaxation with margin')
			legend()
		elif(op==4):
			plot([x1, 0], [0, y1],'--', label = 'Widrow-Hoff')
			legend()
	show()
