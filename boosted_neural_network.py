import tensorflow as tf
import scikeras
from scikeras.wrappers import KerasRegressor
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to get rid of tensor flow warnings
from sklearn.ensemble import AdaBoostRegressor

#importing data
split=200
df=pd.read_csv('TeachingRatings.csv')
data = np.array(df)
xtrain = data[:split,4]
ytrain = data[:split,5]
xtest = data[split:,4]
ytest = data[split:,5]

#creating 'weak' learner as neural network
def boostNN():
    temp=tf.keras.Sequential([
        layers.Dense(4, activation='relu', input_shape=(1,)),
       # layers.Dense(4, activation='linear'),
        layers.Dense(1)
    ])
    temp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='mean_squared_error')
    return temp

#adaptive boosting model
estimator = KerasRegressor(model=boostNN, verbose=0, epochs=800)#deprecation warning on this code
booster = AdaBoostRegressor(base_estimator=estimator, n_estimators=10)
booster.fit(xtrain.reshape(-1,1),ytrain)
bpredict = booster.predict(xtest.reshape(-1,1))

bscore = metrics.mean_squared_error(bpredict,ytest)
print(bscore)

#traditional neural network model
dnn = tf.keras.Sequential([
    layers.Dense(4, activation='relu'),
    #layers.Dense(4, activation='linear'),
    layers.Dense(1)
])
dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='mean_squared_error')
nmodel = dnn.fit(xtrain.reshape(-1,1),ytrain,verbose=0,epochs=800)
npredict = dnn.predict(xtest)
nscore = metrics.mean_squared_error(npredict,ytest)
print(nscore)

#plotting models
xpoints = np.arange(-1.5,2,.1)
bline = booster.predict(xpoints.reshape(-1,1))
nnline = dnn.predict(xpoints)

fic, (ax1,ax2) = plt.subplots(1,2,figsize=(12,16))
ax1.scatter(df['beauty'],df['course_eval'])
ax2.scatter(df['beauty'],df['course_eval'])
ax1.plot(xpoints,bline,label='Boosted')
ax2.plot(xpoints,nnline,label='Neural Network')
plt.show()