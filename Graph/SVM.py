import warnings

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

predict_val = [[1500,390]]

filedata='data.csv'
data1=pd.read_csv(filedata)

X1=data1['1.Time']
X2=data1['2.Signal Value']
X=np.array(list(zip(X1,X2)))
y=data1['5.Health Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

nmodel = GaussianNB()
nmodel.fit(X_train, y_train)
y_pred = nmodel.predict(X_test)
predicted= nmodel.predict(predict_val) # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)
print('accuracy Naive Bayes: ',accuracy_score(y_test, y_pred))

model= RandomForestClassifier()
model.fit(X_train, y_train)
predicted= model.predict(predict_val)
y_pred = model.predict(X_test)
print('random forest', predicted)
print('accuracy random forest: ',accuracy_score(y_test, y_pred))

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred1 = svclassifier.predict(X_test)
op = svclassifier.predict(predict_val)
print('svm', op)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
print('accuracy SVM: ',accuracy_score(y_test, y_pred1))