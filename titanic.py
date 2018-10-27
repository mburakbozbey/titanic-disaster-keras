## Preprocessing of the packages:
import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras import metrics
from keras import backend
import csv

##Reads the data:
train=pd.read_csv("path\\to\\train\\data\\train.csv")
test=pd.read_csv("path\\to\\train\\data\\test.csv")

## Seed the generator to re-seed the generator:
np.random.seed(5)

## In dataset, there are many NaN values for people's ages.
## I set these NaN values with total average of not NaN age
## values.

train = train.fillna(30)
test = test.fillna(30)

## Using data.iloc[<row selection>, <column selection>] syntax,
## to select row and column number:
## Unwanted columns are distracted

x = train.iloc[:,[0,2,4,5,6,7,9]]
y = train.iloc[:,1]
x_test = test.iloc[:,[0,1,3,4,5,6,8]]
x_test_id = test.iloc[:,0]

## Encoding all categorical column values to numeric values:
## i.e. female, male

encoder=LabelEncoder()
for col in x_test.columns.values:
    # Encoding only categorical variables
    if x_test[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
        data=x[col].append(x_test[col])
        encoder.fit(data.values)
        x[col]=encoder.transform(x[col])
        x_test[col]=encoder.transform(x_test[col])

scaler = StandardScaler()
x = scaler.fit_transform(x)             # Compute mean, std and transform training data as well
x_test = scaler.transform(x_test)       # Perform standardization by centering and scaling


## Sequential API of Keras is used.
clf = Sequential()

## Hidden layer:
clf.add(Dense(kernel_initializer = 'uniform',
              input_dim = 7,
              units = 512,
              activation = 'relu'))
clf.add(Dropout(0.5))
## Hidden layer:
clf.add(Dense(kernel_initializer = 'uniform',
              input_dim = 14,
              units = 512,
              activation = 'relu'))
clf.add(Dropout(0.5))
## Output layer:
clf.add(Dense(kernel_initializer = 'uniform',
              units = 1,
              activation = 'sigmoid'))

## Configure the learning process, which is done via the compile method.

rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

clf.compile(optimizer="adam",
            loss='binary_crossentropy',
            metrics=["accuracy"])

## Train the model, iterating on the data in batches of 5 samples:
clf.fit(x,y,
        batch_size=128,
        epochs=1024,
        verbose = 2)
## Prediction with test data.
y_pred = clf.predict(x_test)
## Numpy array is rounded to integer values.
y_pred = np.round(y_pred,0)
## Numpy array is converted to pandas dataframe.
y_pred = pd.DataFrame({'Survived':y_pred[:,0]})
x_test_id.columns = ["PassengerId"]
x_test_id=x_test_id.reset_index()
x_test_id=x_test_id.iloc[:,1]
sub = pd.concat([x_test_id,y_pred], axis=1,ignore_index=True)
sub.columns = ["PassengerId","Survived"]
sub.to_csv("Path\\to\\csv\\file\\submit.csv", sep=',',index=False)


