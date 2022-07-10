import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import itertools
import seaborn as sns
from numpy import linalg as LA


#DATA IMPORT AND VARIABLE CREATION
p=13 # refers to number of X columns being used
credit=pd.read_csv('credit.csv',header=None)
fulldata=np.array(credit)
Y=fulldata[:,24]
X=fulldata[:,0:p]

## performing PCA and using plot to determine optimal number of factors
[n,p]=X.shape
S=X.T@X/n
w, v = LA.eig(S) 
index0=np.argsort(w.real)  
plt.plot(w[index0][::-1]) 
plt.title("eigenvalues")
plt.show()

# creating factors based on plat
K=4 
ED = eigsh(S, k=K, which='LM')
vector = np.flip(ED[1], axis=1)
factor=X@vector/np.sqrt(p) # n by K
m=round(n*0.8)
Yin=Y[0:m]
Yout=Y[m:1000]
#splitting data into holdout samples
factorin= factor[0:m,:]
factorout=factor[m:n,:]


# performing linear regression using factors
factorin1= sm.add_constant(factorin)
lmtmp = sm.OLS(Yin,factorin1)
reg = lmtmp.fit()
factorout1= sm.add_constant(factorout)
ypredPCAout=reg.predict(factorout1)
error=LA.norm(ypredPCAout-Yout)/np.sqrt(len(Yout))
print(error)
 
## FACTOR LASSO
K=4 
ED = eigsh(S, k=K, which='LM')
vector = np.flip(ED[1], axis=1)
factor=X@vector/np.sqrt(p)
common= X@vector@vector.T
U= X-common 

m=round(n*0.8)
factorin= factor[0:m,:]
factorout=factor[m:n,:]
Uin= U[0:m,:]
Uout=U[m:n,:]

## fit data using PCA factors, then find residuals
factorin1= sm.add_constant(factorin)
lmtmp = sm.OLS(Yin,factorin1)
reg = lmtmp.fit()
residualin=Yin-reg.predict(factorin1)
factor_coeff=reg.params

# in-sample lasso regression on residuals
model1 = Lasso(alpha=0.005) #LassoCV(cv)
model1.fit(Uin,residualin)
lasso_coeff=model1.coef_

# out-sample forecast 
factorout1= sm.add_constant(factorout)
z=factorout1@factor_coeff[:, np.newaxis]+ Uout@lasso_coeff[:, np.newaxis]
ypredfactorlassoout=z.flatten()
error=LA.norm(ypredfactorlassoout-Yout)/np.sqrt(len(Yout))
print(error)