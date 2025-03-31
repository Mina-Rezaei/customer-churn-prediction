# Customer Churn Prediction Project

## Project Overview
This project focuses on predicting customer churn using the Telco Customer Churn dataset. We implement and compare multiple machine learning models, including basic and improved XGBoost models, and neural networks using both PyTorch and Keras frameworks. The project also includes an ensemble approach that combines predictions from multiple models for improved accuracy.

## Models Implemented
1. Basic XGBoost Model
2. Improved XGBoost Model (with feature engineering)
3. Neural Network (MLPClassifier)
4. PyTorch Neural Network (with BCEWithLogitsLoss)
5. Keras Neural Network
6. Ensemble Model (Majority Voting)

## Key Files
- `xgboost_churn_model.py`: Basic XGBoost implementation
- `xgboost_churn_model_improved.py`: Enhanced XGBoost with feature engineering
- `neural_network_churn_model.py`: Keras-based neural network
- `pytorch_churn_model.py`: PyTorch-based neural network with BCEWithLogitsLoss
- `minimal_neural_network_churn.py`: MLPClassifier implementation
- `ensemble_model_comparison.py`: Ensemble model implementation and comparison
- `Churn_Prediction_DS_ML_Technical_Notes.md`: Technical documentation

## Model Performance Comparison

| Model | Accuracy | F1 Score | Recall | AUC-ROC |
|-------|----------|----------|---------|---------|
| XGBoost (Basic) | 0.81 | 0.58 | 0.45 | 0.84 |
| XGBoost (Improved) | 0.82 | 0.63 | 0.77 | 0.85 |
| MLPClassifier | 0.79 | 0.61 | 0.74 | 0.83 |
| PyTorch NN (BCEWithLogitsLoss) | 0.76 | 0.60 | 0.73 | 0.83 |
| Keras NN | 0.77 | 0.61 | 0.75 | 0.84 |
| Ensemble (Majority Voting) | 0.80 | 0.62 | 0.76 | 0.85 |

## Ensemble Approach
The project implements an ensemble method that combines predictions from three models:
1. Enhanced XGBoost with feature engineering
2. Keras-based neural network
3. PyTorch-based neural network with BCEWithLogitsLoss

### Ensemble Features
- Majority voting system (requires at least 2 models to agree)
- Individual model probability predictions
- Combined ensemble predictions
- Comprehensive performance comparison
- Visualization of ensemble results

### Key Benefits
1. Reduced false positives and false negatives
2. More robust predictions
3. Better handling of edge cases
4. Improved model stability

## Key Features of PyTorch Implementation
The PyTorch neural network implementation includes several advanced features:
- BCEWithLogitsLoss for better numerical stability in binary classification
- Three-layer architecture (128 → 64 → 32 → 1) with batch normalization and dropout
- Learning rate scheduling with ReduceLROnPlateau
- Optimal threshold finding for balanced precision and recall
- Comprehensive visualization of model performance

## Visualizations
- Learning curves for all models
- ROC curves comparing model performance
- Confusion matrices
- Feature importance plots
- Ensemble comparison plots

## Key Findings
1. Improved XGBoost model shows the best overall performance
2. Neural networks provide competitive results with simpler architecture
3. Feature engineering significantly improves model performance
4. Class imbalance handling is crucial for churn prediction
5. BCEWithLogitsLoss provides stable training for binary classification
6. Ensemble approach provides more robust predictions

## Technical Notes
Detailed technical documentation is available in `Churn_Prediction_DS_ML_Technical_Notes.md`, covering:
- Model optimization techniques
- Deep learning frameworks comparison
- Feature engineering strategies
- Handling imbalanced data
- Model evaluation metrics
- Ensemble methods
- Common interview questions

## Dataset
The project uses the Telco Customer Churn dataset, which includes:
- Customer demographics
- Service subscription details
- Payment information
- Churn status

## Requirements
- Python 3.8+
- Required packages listed in `requirements.txt` and `requirements_ensemble.txt`

## Setup and Usage

### 1. Environment Setup
```bash
# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
pip install -r requirements_ensemble.txt
```

### 2. Data Preparation
```bash
# Place your dataset in the project directory
# Expected filename: telco-customer-churn.csv
```

### 3. Running Models

#### Basic XGBoost
```bash
python xgboost_churn_model.py
```

#### Improved XGBoost
```bash
python xgboost_churn_model_improved.py
```

#### MLPClassifier
```bash
python minimal_neural_network_churn.py
```

#### PyTorch Neural Network
```bash
python pytorch_churn_model.py
```

#### Keras Neural Network
```bash
python neural_network_churn_model.py
```

#### Ensemble Comparison
```bash
python ensemble_model_comparison.py
```

### 4. Viewing Results
- Model metrics are saved in respective output files
- Visualizations are saved as PNG files
- Performance comparisons are saved in CSV files
- Ensemble results are saved in ensemble_outputs/

### 5. Git Setup and Repository Management
```bash
# Initialize Git repository
git init

# Add files to Git
git add README.md
git add *.py
git add *.md
git add requirements.txt
git add requirements_ensemble.txt
git add ensemble_outputs/*

# Create initial commit
git commit -m "Initial commit: Customer Churn Prediction Project"

# Add remote repository (replace with your GitHub repository URL)
git remote add origin https://github.com/yourusername/customer-churn-prediction.git

# Push to GitHub
git push -u origin main
```

## Project Structure
```
customer-churn-prediction/
├── data/
│   └── telco-customer-churn.csv
├── models/
│   ├── xgboost_churn_model.py
│   ├── xgboost_churn_model_improved.py
│   ├── neural_network_churn_model.py
│   ├── pytorch_churn_model.py
│   └── minimal_neural_network_churn.py
├── ensemble/
│   └── ensemble_model_comparison.py
├── outputs/
│   ├── pytorch_model_outputs/
│   ├── xgboost_outputs/
│   └── ensemble_outputs/
├── requirements.txt
├── requirements_ensemble.txt
├── README.md
└── Churn_Prediction_DS_ML_Technical_Notes.md
```

## Contributing
Feel free to submit issues and enhancement requests!
