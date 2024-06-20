# Decision Trees for Machine Learning / Predicting Student Academic Success

## Overview

This repository contains the code for a project aimed at predicting student academic success using Decision Trees. The analysis utilizes data from a Portuguese University, covering academic, demographic, and socio-economic variables. The project focuses on different configurations of Decision Trees, including pruning and bagging techniques, to enhance model performance and identify key predictors of student success.

## Problem Statement

The objective of this project is to analyze the dataset of university students and develop predictive models to classify students into three categories of academic success: graduated on time, graduated with a delay, or did not graduate. By identifying key factors contributing to these outcomes, the project aims to provide insights for potential interventions to improve student retention and success rates.

### Model Training and Evaluation

The project employs various configurations of Decision Tree classifiers:

1. **Single Decision Tree**: A basic Decision Tree model.
2. **Pruned Decision Tree**: Hyperparameters optimized using `RandomizedSearchCV` to prevent overfitting.
3. **Bagging with a Single Decision Tree**: Bagging technique to improve model stability.
4. **Bagging with a Pruned Decision Tree**: Combines pruning and bagging for enhanced performance.

### Cross-Validation

The `evaluate_model` function performs 10-fold cross-validation and evaluates the models on the test set, returning various metrics and their cross-validation scores.

### Key Findings
The project evaluated four configurations of Decision Tree classifiers across three data balancing techniques (unbalanced, oversampled, and undersampled). Here are the key findings:
### Single Decision Tree:

- **Unbalanced Data**:
  - Moderate performance
  - Lower recall and F1 scores
  - Model struggles to correctly identify all instances of student success categories

- **Oversampling**:
  - Significantly improves recall and F1 scores
  - Better identification of student success across all categories

- **Undersampling**:
  - Lowest performance
  - Reduced dataset size affects the model's ability to generalize

### Pruned Decision Tree:

- **General Improvements**:
  - Improved accuracy and Kappa scores compared to the single decision tree model

- **Oversampled Data**:
  - Improved recall and F1 scores
  - Better generalization across a balanced dataset

- **Undersampled Data**:
  - Moderate performance
  - Better than the single decision tree model but still not ideal

### Bagged Trees:

- **General Enhancements**:
  - Enhanced stability and overall performance

- **Oversampled Data**:
  - Highest performance.
  - Bagging with oversampling is highly effective

- **Unbalanced Data**:
  - Performs well, though less effective with undersampled data

### Bagged and Pruned Trees:

- **Combined Techniques**:
  - Balanced improvement in accuracy, recall, and F1 scores across all data configurations

- **Oversampled Data**:
  - High performance
  - Effectiveness of combining both techniques

- **Unbalanced and Undersampled Data**:
  - Good performance, but not as high as with oversampled data


## Conclusion

This project demonstrates an approach to predicting student academic success using Decision Tree and highlights the impact of different techniques like pruning and bagging on model performance.
