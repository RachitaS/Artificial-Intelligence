#!/usr/bin/env python
#coding: utf-8

import numpy as np

def generateData(k):
    inputs = []
    x1 = [1,7,8,9,4,8]
    y1 = [6,2,9,9,8,5]
    x2 = [-2,-3,-2,-7,-1,-5]
    y2 = [-1,-3,-4,-1,-3,-2]
    n = len(x1)
    inputs.extend([[x1[i], y1[i], 1, 1] for i in range(n)])
    inputs.extend([[x2[i], y2[i], -1, -1] for i in range(n)])

    test = []
    x1 = [3,8,8]
    y1 = [5,3,1]
    x2 = [-4,-4,-5]
    y2 = [-1,-3,0]
    n = len(x1)
    test.extend([[x1[i], y1[i], 1, 1] for i in range(n)])
    test.extend([[x2[i], y2[i], -1, -1] for i in range(n)])

    if k==1:
    	return inputs
    else:
    	return test
