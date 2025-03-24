# Customer Churn Prediction Project

This repository contains machine learning models to predict customer churn using the Telco Customer Churn dataset. Multiple approaches are implemented and compared, with a focus on handling imbalanced data.

## Project Overview

Customer churn prediction is a critical business problem, especially in subscription-based services. This project demonstrates how to build increasingly sophisticated models to improve recall (detection of customers likely to churn) while maintaining good overall performance.

## Models Implemented

### 1. Basic XGBoost Model
- Standard preprocessing
- Default threshold (0.5)
- No class imbalance handling

### 2. Improved XGBoost Model
- Advanced feature engineering
- SMOTE oversampling
- Optimized threshold
- Hyperparameter tuning

### 3. Neural Network Models
- **TensorFlow/Keras**: Deep neural network with dropout, batch normalization, and L1/L2 regularization
- **Scikit-learn MLPClassifier**: Simpler implementation with two hidden layers

## Key Files

- `neural_network_churn_model.py`: TensorFlow implementation
- `minimal_neural_network_churn.py`: Scikit-learn MLPClassifier implementation
- `Churn_Prediction_DS_ML_Technical_Notes.md`: Comprehensive technical notes covering ML concepts and interview questions

## Model Performance Comparison

| Metric | XGBoost Original | XGBoost Improved | Neural Network | MLPClassifier |
|--------|------------------|------------------|----------------|---------------|
| Accuracy | 81.62% | 78.85% | ~80%* | 71.75% |
| F1 Score (Churn) | 0.61 | 0.66 | ~0.67* | 0.58 |
| Recall (Churn) | 55% | 77% | ~75%* | 74% |
| AUC-ROC | Not calculated | 0.858 | ~0.86* | 0.785 |
| Class balancing technique | None | SMOTE | SMOTETomek + Class Weights | SMOTE |

*Values for TensorFlow Neural Network may vary slightly with each run

## Visualizations

### Neural Network Learning Curve (MLPClassifier)
![MLPClassifier Learning Curve](mlp_learning_curve.png)

### ROC Curves
![ROC Curve for MLPClassifier](mlp_roc_curve.png)

### Confusion Matrix
![Confusion Matrix for MLPClassifier](mlp_confusion_matrix.png)

## Key Findings

1. The Improved XGBoost model achieved the best recall (77%) for churning customers, making it most valuable from a business perspective.
2. The TensorFlow Neural Network model provided comparable performance with potentially better ability to capture complex patterns.
3. The MLPClassifier offers a simpler neural network implementation with reasonable performance.
4. All advanced models significantly outperformed the baseline in terms of recall for the minority class.

## Technical Notes

This repository includes comprehensive technical notes (`Churn_Prediction_DS_ML_Technical_Notes.md`) covering:

- Machine learning fundamentals
- Feature engineering techniques
- Handling imbalanced data
- XGBoost deep dive
- Neural networks for imbalanced classification
- Model evaluation metrics
- Common interview questions and answers

## Dataset

The project uses the Telco Customer Churn dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

## Requirements

- pandas
- numpy
- scikit-learn
- imbalanced-learn 
- matplotlib
- seaborn
- tensorflow (for the deep neural network model)

## Setup and Usage

1. Clone this repository
2. Install the required packages
3. Download the dataset and place it in the project directory
4. Run the desired model script

```bash
# For TensorFlow neural network model
python neural_network_churn_model.py

# For scikit-learn MLPClassifier
python minimal_neural_network_churn.py
``` 