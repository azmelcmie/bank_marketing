# Artificial Neural Network
Python code using [Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) to predict whether a bank customer will subscribe to a term deposit.

## Input & Output
The input comes in the form of a comma separated value file showing various details of the bank's customers. Not all fields will need to be used in this project.

Below is a sample of the data.

![bank_additional_full](https://github.com/azmelcmie/bank_marketing/blob/master/img/bm_dataset.PNG)

 
The results can be viewed in the form of a confusion matrix that will show how many people will subscribe to the bank's term deposit (the figure in blue), how many will decline along with false positives and negatives.

![cm_ann](https://github.com/azmelcmie/bank_marketing/blob/master/img/cm_ann.PNG)

## Environment
Coded and tested in Anaconda version 5.2.0 using the Spyder 3.2.8 environment.

### Libraries

Library | Version
--------| -----------
numpy | 1.14.3
matplotlib | 2.2.2
pandas | 0.23.0
scikit-learn | 0.19.1
tensorflow | 1.6.0
keras | 2.2.4


## Usage
Run the **bm_ann.py** file from within the Spyder environment. Select all code and press CTRL-ENTER to run the program. Double-click on confusion matrix (cm) in Variable explorer to view the confusion matrix table.

To experiment with your own data, you can edit the following lines.

Edit the location and name of your dataset along with the required columns (**lines 7-12**):

````python
dataset = pd.read_csv('data/bank_additional_full.csv')
# Choosing the relevent rows, columns for independent variable
# np.r_ used for translating slice object
X = dataset.iloc[:, np.r_[0:7,14:20]].values
# Choosing the relevent rows, columns for the dependent variable
y = dataset.iloc[:, 20].values
````

Experiment with the layer parameters (**lines 63-72**):

````python
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=35, units=10))

# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=10))

# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compiling the ANN. Stochastic Gradient Descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
````

Experiment also with the batch size and number of epochs (**line 76**):

````python
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
````

## Citations
Dataset: *bank_additional_full.csv*.

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
