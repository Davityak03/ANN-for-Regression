# ANN-for-Regression

## Project Overview

This project involves developing an Artificial Neural Network (ANN) to perform regression analysis on the `Folds5x2_pp.csv` dataset. The objective is to predict the energy output of a Combined Cycle Power Plant based on various environmental parameters.

## Dataset

The `Folds5x2_pp.csv` dataset contains five columns:

1. **AT**: Temperature (in °C)
2. **V**: Exhaust Vacuum (in cm Hg)
3. **AP**: Ambient Pressure (in millibar)
4. **RH**: Relative Humidity (in %)
5. **PE**: Net hourly electrical energy output (in MW) - Target variable

The dataset consists of 9568 instances and is publicly available for research purposes.

## Prerequisites

To run this project, you'll need the following software and libraries installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib 

## Project Structure

The project directory contains the following files:

- `Folds5x2_pp.csv`: The dataset file.
- `ann_regression.py`: The main script to preprocess data, build, train, and evaluate the ANN model.
- `README.md`: This README file.

## Usage

1. **Data Preprocessing**: The script `ann_regression.py` includes data preprocessing steps such as loading the dataset, handling missing values (if any), and splitting the data into training and testing sets.

2. **Building the ANN Model**: The script constructs an ANN model using the Keras library. The model consists of:
   - Input layer: Corresponding to the number of features (4 in this case).
   - Hidden layers: Two hidden layers with a specified number of neurons and activation functions.
   - Output layer: A single neuron for regression output.

3. **Training the Model**: The ANN model is trained on the training dataset using mean squared error (MSE) as the loss function and Adam optimizer.

4. **Evaluation**: The trained model is evaluated on the testing dataset to measure its performance. The script calculates and prints the mean absolute error (MAE) and root mean squared error (RMSE).

5. **Visualization **: The script can plot the training and validation loss over epochs if Matplotlib is installed.
