from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import scipy
import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import svm

# classification problem using randomized data
n=50
X=np.random.normal(0,1,(n,2))
Y0=X@np.array([0.5,-1])+np.random.normal(0,1,(n,1)).flatten()
Y=Y0>0   

#decision tree classifier
clf = tree.DecisionTreeClassifier(max_leaf_nodes= 5)
res = clf.fit(X, Y)
ypredtree = res.predict(X)
tree.plot_tree(res)

#  random forest model with 1000 decision trees
rfRF = RandomForestClassifier(n_estimators = 1000)
resRF=rfRF.fit(X, Y)
ypredforest= resRF.predict(X)  

#svm classifier
svmy= 1*Y     
svmmodel = svm.SVC(kernel='linear',C=100)
svmmodel.fit(X, svmy)
b=svmmodel.coef_[0]
a=svmmodel.intercept_[0]
linex = np.arange(np.min(X[:,0]),np.max(X[:,0]),.1)
liney= -(b[0] / b[1]) * linex - a / b[1]

#plotting comparison of first 3 models
fig, (axs1,axs2,axs3) = plt.subplots(1,3,figsize=(12,16))
axs1.scatter(X[Y==0,0],X[Y==0,1],color="black",marker="+", label="Y=0" )
axs1.scatter(X[Y==1,0],X[Y==1,1],color="blue", label="Y=1" )
axs1.set(ylabel='X2',title="true")
axs1.plot(linex,liney, label="SVM line")
axs1.legend(loc="lower right")

axs2.scatter(X[ypredtree==0,0],X[ypredtree==0,1],color="black",marker="+", label="Ypredict=0" )
axs2.scatter(X[ypredtree==1,0],X[ypredtree==1,1],color="blue", label="Ypredict=1" )
axs2.set(xlabel='X1', ylabel='X2',title="tree")
axs2.legend(loc="lower right")


axs3.scatter(X[ypredforest==0,0],X[ypredforest==0,1],color="black",marker="+", label="Ypredict=0" )
axs3.scatter(X[ypredforest==1,0],X[ypredforest==1,1],color="blue", label="Ypredict=1" )
axs3.set(ylabel='X2',title="random forests")
axs3.legend(loc="lower right")


# regression problem with randomized data
n=100
X=np.random.normal(0,1,(n,1))
e=np.random.normal(0,1,(n,1))
Y=  X-X**2+e
plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
x_points=np.arange(-2,3,0.1)
y_points=x_points-x_points**2

# tree    
clf = tree.DecisionTreeRegressor(max_leaf_nodes= 10)
res = clf.fit(X, Y.flatten())
ypredtree = res.predict(x_points[:, np.newaxis])

# random forest 
rfRF = RandomForestRegressor(n_estimators = 1000)
resRF=rfRF.fit(X, Y.flatten())
ypredforest= resRF.predict(x_points[:, np.newaxis])  

#plotting true vs. tree vs. random forest
plt.plot(x_points,y_points,color='b',linestyle='solid',label="true")
plt.plot(x_points,ypredtree,color='k',linestyle='solid', label="tree")
plt.plot(x_points,ypredforest,color='r',linestyle='solid', label="forest")
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()