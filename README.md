# Diabetes Linear Regression
This project demonstrates the use of Linear Regression to predict disease progression based on the diabetes dataset. The project is implemented in Python using the scikit-learn library.  
## Project Structure
main.py: The main script that performs the following tasks:
Loads the diabetes dataset.
Inspects the structure of the dataset.
Splits the dataset into training and test sets.
Trains a Linear Regression model.
Makes predictions on the test set.
Evaluates the model using Mean Squared Error (MSE).

Explanation of Tasks
Load the Dataset:  
The diabetes dataset is loaded using load_diabetes from scikit-learn.
Inspect the Structure:  
The feature matrix and target variable are extracted and their structure is inspected.
Split the Dataset:  
The dataset is split into training and test sets using train_test_split.
Train the Model:
A Linear Regression model is trained using the training data.
Make Predictions:  
Predictions are made on the test set and compared to the actual values using a scatter plot.
Evaluate the Model:  
The model is evaluated using Mean Squared Error (MSE).
