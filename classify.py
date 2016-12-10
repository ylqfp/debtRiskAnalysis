#!/bin/python
#-*- coding:utf-8 -*-

import sys
import re
import numpy as np
import sklearn
from sklearn import tree, svm
import random
from IPython.display import Image 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsemble
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
#from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import train_test_split

fin=open("data.csv", 'r')
fout=open("res.txt", 'w')
dic = {}

fin.readline()

lst0 = []
lst1 = []
lst2 = []

X_train = []
Y_train = []

X_test = []
Y_test = []

def makeData():
	#sm = SMOTE(kind='svm')
	#X_resampled, y_resampled = sm.fit_sample(X, y)
	global X_train,Y_train,X_test,Y_test
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	l0 = int(len(lst0)*0.8)
	l1 = int(len(lst1)*0.8)
	l2 = int(len(lst2)*0.8)
	# label 0
	ind = range(len(lst0))
	t1 = random.sample(ind, l0)
	t2 = list(set(ind) - set(t1))
	for i in t1:
		X_train.append(lst0[i])
		Y_train.append(0)
	for i in t2:
		X_test.append(lst0[i])
		Y_test.append(0)
	# label 1
	ind = range(len(lst1))
	t1 = random.sample(ind, l1)
	t2 = list(set(ind) - set(t1))
	for i in t1:
		X_train.append(lst1[i])
		Y_train.append(1)
		#for j in range(5):# add fake data
		#	nl = [x + random.uniform(-0.1,0.1) for x in lst1[i]]
		#	X_train.append(nl)
		#	Y_train.append(1)
	for i in t2:
		X_test.append(lst1[i])
		Y_test.append(1)
		#for j in range(50):# add fake data
		#	nl = [x + random.uniform(-1,1) for x in lst1[i]]
		#	X_test.append(nl)
		#	Y_test.append(1)
	# label 2
	ind = range(len(lst2))
	t1 = random.sample(ind, l2)
	t2 = list(set(ind) - set(t1))
	for i in t1:
		X_train.append(lst2[i])
		Y_train.append(2)
		#for j in range(50):# add fake data
		#	nl = [x + random.uniform(-0.1,0.1) for x in lst2[i]]
		#	X_train.append(nl)
		#	Y_train.append(2)
	for i in t2:
		X_test.append(lst2[i])
		Y_test.append(2)
	print "TrainSet %d, TestSet %d." % (len(X_train), len(X_test))
	print "Each sample %d len" % (len(X_train[0]))
	#print X_train[:3], Y_train[:3]
	#print X_test[:3], Y_test[:3]

	#sm = SMOTE(kind='svm')
	#X_resampled, Y_resampled = sm.fit_sample(X_train, Y_train)
	ee = EasyEnsemble()
	X_resampled, Y_resampled = ee.fit_sample(X_train, Y_train)
	print "After resampled, TrainSet %d, TestSet %d." % (len(X_train), len(X_test))
	#X_train = X_resampled
	#Y_train = Y_resampled
	print Y_train[:10]
	print Y_resampled[:10]

	
	
badline = 0

def checkLine(features):
	res = -1
	for cell in features:
		c = cell.strip()
		if c == None:
			return res
		try:
			if not int(c)>=0:
				return res
		except Exception, e:
			return res
	return 1
	
def train():
	global X_train, X_test, Y_train, Y_test
	finalresult = 0
	nIters = 1
	target_names = ['class 0', 'class 1', 'class 2']
	for i in range(nIters): # 训练10次，取最后结果的平均值，每次80/20
		makeData()
		#clf = tree.DecisionTreeClassifier() #DecisionTreeClassifier
		clf = OneVsRestClassifier(estimator = SVC(random_state=0, probability=True)) #SVM
		#clf = RandomForestClassifier(n_estimators=70) # Random Forest
		#clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)

		#clf = clf.fit(np.array(X_train), np.array(Y_train))
		#res = clf.score(np.array(X_test), np.array(Y_test))
		clf = clf.fit(X_train, Y_train)
		res = clf.score(X_test, Y_test)		
		Y_pred = clf.predict_proba(X_test)
		Y_pred1 = []
		for prob_tuple in Y_pred:
			a = prob_tuple[0]
			b = prob_tuple[1]
			c = prob_tuple[2]
			if a>=b and a>=c:
				Y_pred1.append(0)
			elif b>=a and b>=c:
				Y_pred1.append(1)
			elif c>=a and c>=b:
				Y_pred1.append(2)
		print Y_pred1
		print Y_test
		#X1 = np.array(X_train)
		#X2 = np.array(X_test)
		#Y1 = np.array(Y_train)
		#Y2 = np.array(Y_test)
		#print X1.shape, Y1.shape
		#clf = clf.fit(X1, Y1)
		#res = clf.score(X2, Y2)		
		print res
		finalresult += res
		print classification_report(Y_test, Y_pred1, target_names=target_names)
	print "final Result: %.4f" % (finalresult/nIters)
		
for line in fin:
	lines = line.strip().split(",")
	label = lines[0]
	zhuti = lines[1]
	features = lines[2:8]
	f = "".join(features)
	if checkLine(features) == -1:
		continue
	try:
		features_int = [int(x) for x in features]
	except Exception, e:
		print e
		badline += 1
		continue
	#print features
	if label == "0":
		lst0.append(features_int)
	elif label == "1":
		lst1.append(features_int)
	elif label == "2":
		lst2.append(features_int)
print "%d label0, %d label1, %d label2" % (len(lst0), len(lst1), len(lst2))
print "Total %d bad lines" % badline

train()
