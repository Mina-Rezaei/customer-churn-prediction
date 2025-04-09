import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('telco-customer-churn.csv')

# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, handling empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values in TotalCharges with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert Churn to binary (Yes=1, No=0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Hyperparameter tuning with cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

# Train final model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate final model
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\nFinal Model Accuracy: {accuracy_best:.4f}")
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_best))

# Plot feature importance of the best model
plt.figure(figsize=(10, 6))
feature_importance_best = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance_best = feature_importance_best.sort_values('importance', ascending=False)
sns.barplot(x='importance', y='feature', data=feature_importance_best)
plt.title('Final Model Feature Importance')
plt.tight_layout()
plt.savefig('final_feature_importance.png')
plt.close()

# Save the model
best_model.save_model('xgboost_churn_model.json')
print("\nModel saved as 'xgboost_churn_model.json'") 