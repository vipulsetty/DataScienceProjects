import scipy
from scipy.sparse.linalg import eigsh
import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

papers=pd.read_csv('Federalist.csv',header=None)

fulldata=np.array(papers)
print(fulldata)
Y=fulldata[:,0]

col=5

X=fulldata[:,1:col]
m=32; # in sample size
[n,p]=X.shape
S=X.T@X/n
w, v = LA.eig(S)   # w = value, v= vector
index0=np.argsort(w.real)  # index from min to max
plt.plot(w[index0][::-1]) 
plt.title("eigenvalues")

#yuan pca copy
K=4 # num factors
ED = eigsh(S, k=K, which='LM')
vector = np.flip(ED[1], axis=1)
factor=X@vector/np.sqrt(p) # n by K
factorin= factor[0:m,:]
factorout=factor[m:44,:]
Xin=X[0:m,:]
Yin=Y[0:m]
Xout=X[m:44,:] # predict
Yout=Y[m:44] # disputed, unknown,

#pca regression
model = LogisticRegression().fit(factorin,Yin)
ypredict = model.predict(factorin)
ypredict = (ypredict>.5)+0

score = metrics.accuracy_score(ypredict,Yin)
print(score)

plt.show()