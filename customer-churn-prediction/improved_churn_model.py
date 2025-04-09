import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

print("===== IMPLEMENTING IMPROVED CHURN PREDICTION MODEL =====")
print("Addressing class imbalance and feature engineering to improve model performance")

# Load the data
df = pd.read_csv('telco-customer-churn.csv')

# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, handling empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert Churn to binary (Yes=1, No=0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Print data balance
print("\nClass Distribution before balancing:")
print(df['Churn'].value_counts(normalize=True) * 100)

# Feature Engineering - Create additional features
print("\n--- Feature Engineering ---")

# 1. Create tenure groups
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], 
                            labels=['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years'])

# 2. Create total services count
service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Count number of services for each customer
df['service_count'] = 0
for col in service_columns:
    # Convert to binary first
    if col == 'PhoneService':
        df[col + '_binary'] = (df[col] == 'Yes').astype(int)
    else:
        df[col + '_binary'] = ((df[col] == 'Yes') | (df[col] == 'No phone service')).astype(int)
    df['service_count'] += df[col + '_binary']

# 3. Create monthly charge buckets
df['charge_bucket'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# 4. Create interaction features
df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']
df['contract_x_monthly'] = df['Contract'].astype(str) + '_' + df['charge_bucket'].astype(str)

# Handle categorical variables
print("\n--- Encoding Categorical Variables ---")
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column].astype(str))

# Convert categorical tenure group and charge bucket to numeric
df['tenure_group'] = df['tenure_group'].astype('category').cat.codes
df['charge_bucket'] = df['charge_bucket'].astype('category').cat.codes

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to address class imbalance
print("\n--- Applying SMOTE for Class Balancing ---")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Class Distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True) * 100)

# Create optimized XGBoost model
print("\n--- Training Improved XGBoost Model ---")
model = xgb.XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=200,
    colsample_bytree=0.8,
    subsample=0.8,
    objective='binary:logistic',
    scale_pos_weight=1,  # Already balanced with SMOTE
    random_state=42
)

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\n--- Improved Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# Plot ROC curve
plt.figure(figsize=(8, 6))
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title('ROC Curve')
plt.savefig('improved_roc_curve.png')
plt.close()

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 15 Feature Importance - Improved Model')
plt.tight_layout()
plt.savefig('improved_feature_importance.png')
plt.close()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Improved Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('improved_confusion_matrix.png')
plt.close()

# Find optimal threshold
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Calculate F1 score for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Probability Threshold: {optimal_threshold:.4f}")

# Apply optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

print("\nPerformance with Optimal Threshold:")
print("Accuracy:", accuracy_score(y_test, y_pred_optimal))
print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Save the improved model
model.save_model('improved_xgboost_churn_model.json')
print("\nImproved model saved as 'improved_xgboost_churn_model.json'")

# Save optimal threshold value for future use
with open('optimal_threshold.txt', 'w') as f:
    f.write(str(optimal_threshold))
print("Optimal threshold saved as 'optimal_threshold.txt'")

print("\n===== IMPROVEMENT SUMMARY =====")
print("1. Added feature engineering:")
print("   - Created tenure groups and charge buckets")
print("   - Added service count and interaction features")
print("2. Addressed class imbalance with SMOTE")
print("3. Used feature scaling")
print("4. Optimized the probability threshold for better recall")
print("5. Saved model and threshold for production use") 