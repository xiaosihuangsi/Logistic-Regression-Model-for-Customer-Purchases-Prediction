# Logistic-Regression-Model-for-Customer-Purchases-Prediction


## Overview

This project implements a logistic regression model to predict customer purchases based on demographic information such as age, gender, and estimated salary. The model is trained on a dataset and evaluated using test data. Additionally, a single input prediction is demonstrated.

## Project Structure

- **`sales_data.csv`**: Dataset containing information about customer purchases and demographic data.
- **`week_36_exercise1_logistic_regression.py`**: Python script implementing the logistic regression model, data preprocessing, training, testing, and single input prediction.

## Dependencies

This project requires the following libraries:

- `seaborn`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/customer-purchases-prediction.git
    cd customer-purchases-prediction
    ```

2. **Activate the Anaconda environment:**

    ```bash
    conda activate ml
    ```

3. **Run the script:**

    ```bash
    python week_36_exercise1_logistic_regression.py
    ```

    The script will load the dataset, preprocess the data, train the logistic regression model, and evaluate its performance on test data. It will also demonstrate a single input prediction.


## Results

The script generates a confusion matrix heatmap to visualize the performance of the logistic regression model on test data. Additionally, the accuracy of the model is calculated. The single input prediction demonstrates how to use the trained model for making predictions with new data.

