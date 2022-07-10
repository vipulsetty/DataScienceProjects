import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('credit.csv',header=None)
total=np.array(df)

split = 500
columns = 15
ytrain = total[:split,24]-1
ytest = total[split:,24]-1
xtrain = total[:split,:columns]
xtest = total[split:,:columns]

model = LogisticRegression(fit_intercept=True, solver='liblinear').fit(xtrain, ytrain)
ypredict = model.predict(xtest)
yfitted = model.predict(xtrain)
accuracy1=metrics.accuracy_score(ytrain, yfitted)
accuracy=metrics.accuracy_score(ytest, ypredict)
yfitted = model.predict(xtrain)
print(accuracy1)
print(accuracy)

#logostic regression plotting
fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2,2,figsize=(12,16))
axs1.scatter(xtrain[ytrain==0,9],xtrain[ytrain==0,1],color="black",marker="+", label="Y=0" )
axs1.scatter(xtrain[ytrain==1,9],xtrain[ytrain==1,1],color="blue", label="Y=1" )
axs2.scatter(xtrain[ytrain==yfitted,9], xtrain[ytrain==yfitted,1],color="green",marker="+", label="Correct" )
axs2.scatter(xtrain[ytrain!=yfitted,9],xtrain[ytrain!=yfitted,1],color="red", label="incorrect" )
axs3.scatter(xtest[ytest==0,9],xtest[ytest==0,1],color="black",marker="+", label="Y=0" )
axs3.scatter(xtest[ytest==1,9],xtest[ytest==1,1],color="blue", label="Y=1" )
axs4.scatter(xtest[ytest==ypredict,9], xtest[ytest==ypredict,1],color="green",marker="+", label="Correct" )
axs4.scatter(xtest[ytest!=ypredict,9],xtest[ytest!=ypredict,1],color="red", label="incorrect" )
plt.show()

#NEURAL NETWORKS
max_epo=500
lr=0.1

dnn = tf.keras.Sequential([
  #  normalizer,
      layers.Dense(8, activation='relu'),
      layers.Dense(4, activation='relu'),
      layers.Dense(1, activation=tf.nn.sigmoid)
])

dnn.compile(tf.keras.optimizers.Adadelta(learning_rate=lr),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history=dnn.fit(xtrain, ytrain,epochs=max_epo,verbose=0)
ynnpredict = dnn.predict(xtest)
ynnpredict = (ynnpredict>.5).astype(int) 
accuracynn=metrics.accuracy_score(ytest, ynnpredict)
print(accuracynn)