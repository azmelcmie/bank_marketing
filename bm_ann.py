# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/bank_additional_full.csv')
# Choosing the relevent rows, columns for independent variable
# np.r_ used for translating slice object
X = dataset.iloc[:, np.r_[0:7,14:20]].values
# Choosing the relevent rows, columns for the dependent variable
y = dataset.iloc[:, 20].values

# Encoding categorical data
# Using LabelEncoder to convert text to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])

# Using OneHotEncoder to split LabelEncoder data into 
# multiple columns to avoid any supposed hierarchy order 
onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7])
X = onehotencoder.fit_transform(X).toarray()
# Removing dummy variables
X = X[:, np.r_[1:12, 13:16, 17:24, 25:27, 28:30, 31:33, 34:42]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling to standardise the range of independent variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using ANN
# Importing the Keras libraries and packages
import keras
# Initialize the neural network
from keras.models import Sequential
# The model used for creating the layers in ANN
from keras.layers import Dense

# Initialising the ANN
# Sequence of layers. Classification problem
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Best to experiment with the parameters
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=35, units=10))

# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=10))

# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compiling the ANN. Stochastic Gradient Descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Using 100 rounds, 10 at a time
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results with a threshold of 50%
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)