# deep-learning-challenge
## Report
After creating and tuning the nn model, here are my takeaways on the model's performance and the tuning process.

## Overview
This analysis will cover the details of the neural network model created for this challenge. Further, the adjustments made to the model in order to increase performance will be described.

## Results
Data Preprocessing

The target variable in this model is the "IS_SUCCESSFUL" column.

The feature variables in the final model were:
NAME
APPLICATION_TYPE
AFFILIATION
CLASSIFICATION
USE_CASE
ORGANIZATION
INCOME_AMT
ASK_AMT
The variable that was removed from the final model were:
STATUS - this column had over 34k 1's and only 5 0's - only five of the rows are zeroes which is not helpful.

## Compiling, Training, and Evaluating the Model
The final model had three layers. The first layer had 100 neurons or nodes, the second layer had 30 neurons, and the third layer had 10 neurons. The model used two different activation functions - rectified linear units (relu) and sigmoid. 
![image](https://github.com/kanienie/deep-learning-challenge/assets/124482339/d54a2974-fbf9-44ea-8bc7-e5e1c6f67a8b)

## model definition and summary

The model exceeded the target performance of 75%. The model reached 79% accuracy 
![image](https://github.com/kanienie/deep-learning-challenge/assets/124482339/5d16e632-eeb0-4fd8-b2ef-c8acbbe26c28)


I took the following steps to increase the model's performance:
Including the NAME column. There were a significant number of organization names that were repeated in the data. I thought there could be a connection betwen the organization and its success.
Two columns, SPECIAL_CONSIDERATIONS and STATUS were removed because they were both almost all one value.
I added a third layer to increase the model's complexity.
I changed the activation functions on the second and third layers as well as the output layer.

## Summary
The tuned neural network was able to predict outcomes with 79% accuracy. This is a marked increase when compared to the original neural network which predicted outcomes with 73% accuracy. This was achieved by adding the organization NAME to the feature variables, removing SPECIAL_CONSIDERATIONS and STATUS from the feature variables, increasing the model's complexity by adding a third hidden layer, and changing some of the layers' activation functions.

As an alternative to the nn model, a random forest classifier could be used. I attempted this and was able to get 78% accuracy. This is a much better score than the 75% threshold I was shooting for. However, the neural network won out with 79% accuracy. With further tuning, it may be possible to improve on one or both of these models and get an even better predictive classifier.


## Instructions
## Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

## Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
