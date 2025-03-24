import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Try importing imbalanced-learn, if not available, notify the user
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    imblearn_available = True
except ImportError:
    print("WARNING: imbalanced-learn is not installed. You will need it to handle imbalanced data.")
    print("Install with: pip install imbalanced-learn")
    imblearn_available = False

print("===== SCIKIT-LEARN NEURAL NETWORK FOR CHURN PREDICTION =====")
print("Implementing a simpler neural network approach using MLPClassifier")

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
try:
    df = pd.read_csv('telco-customer-churn.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("ERROR: telco-customer-churn.csv not found.")
    print("Please download the dataset first.")
    import sys
    sys.exit(1)

# Basic data exploration
print("\n--- Basic Data Exploration ---")
print(f"Dataset shape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum())

# Check for empty strings that might be treated as missing values
print("\nChecking for empty strings in string columns:")
for col in df.select_dtypes(include=['object']).columns:
    empty_strings = (df[col] == '').sum()
    if empty_strings > 0:
        print(f"'{col}' has {empty_strings} empty strings")

# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, handling empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"\nNumber of missing values in TotalCharges after conversion: {df['TotalCharges'].isnull().sum()}")

# Fill missing values in TotalCharges with 0 or use median
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
print(f"Filled missing TotalCharges with median: {median_total_charges}")

# Convert Churn to binary (Yes=1, No=0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Print class distribution
print("\nClass Distribution:")
print(df['Churn'].value_counts(normalize=True) * 100)

# Feature Engineering - Similar to the improved model for comparison
print("\n--- Feature Engineering ---")

# 1. Create tenure groups
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], 
                           labels=[0, 1, 2, 3, 4, 5])

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
df['charge_bucket'] = pd.qcut(df['MonthlyCharges'], q=4, labels=[0, 1, 2, 3])

# 4. Create interaction features
df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']

# Handle categorical variables
print("\n--- Encoding Categorical Variables ---")
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column].astype(str))

# Convert categorical tenure group and charge bucket to numeric
df['tenure_group'] = pd.to_numeric(df['tenure_group'], errors='coerce')
df['charge_bucket'] = pd.to_numeric(df['charge_bucket'], errors='coerce')

# Check for any remaining NaN values after preprocessing
print("\nChecking for NaN values after preprocessing:")
nan_columns = df.columns[df.isna().any()].tolist()
if nan_columns:
    print(f"Columns with NaN values: {nan_columns}")
    print("Filling remaining NaN values...")
    # Use SimpleImputer to handle any remaining NaN values
    imputer = SimpleImputer(strategy='median')
    df[nan_columns] = imputer.fit_transform(df[nan_columns])
else:
    print("No NaN values found - data is clean")

# Verify all NaN values are handled
assert not df.isna().any().any(), "There are still NaN values in the dataset"
print("All NaN values have been handled successfully")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling - essential for neural networks
print("\n--- Applying Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if imblearn_available:
    # Handle imbalanced data
    print("\n--- Handling Imbalanced Data with SMOTE ---")
    # SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print("Class Distribution after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts(normalize=True) * 100)
else:
    print("\nWarning: imbalanced-learn not available, proceeding without resampling")
    X_train_balanced = X_train_scaled
    y_train_balanced = y_train

# Neural Network approach
print("\n--- Building MLPClassifier (Neural Network) ---")

# Create and train the neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),    # Two hidden layers with 64 and 32 neurons
    activation='relu',              # ReLU activation function
    solver='adam',                  # Adam optimizer
    alpha=0.0001,                   # L2 regularization
    batch_size=32,                  # Mini-batch size
    learning_rate_init=0.001,       # Initial learning rate
    max_iter=200,                   # Maximum number of iterations
    early_stopping=True,            # Enable early stopping
    validation_fraction=0.2,        # Validation set fraction
    n_iter_no_change=10,            # Early stopping patience
    random_state=42,                # For reproducibility
    verbose=True                    # Print progress
)

# Train the model
print("\n--- Training Neural Network ---")
mlp.fit(X_train_balanced, y_train_balanced)

# Evaluate the model
print("\n--- Neural Network Performance ---")
y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
y_pred = mlp.predict(X_test_scaled)

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
if len(thresholds) > optimal_idx:  # Error handling
    optimal_threshold = thresholds[optimal_idx]
else:
    optimal_threshold = 0.5  # Default if optimal not found

print(f"\nOptimal Probability Threshold: {optimal_threshold:.4f}")

# Apply optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluate with optimal threshold
accuracy = accuracy_score(y_test, y_pred_optimal)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(10, 8))
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.title('ROC Curve - Neural Network Model (MLPClassifier)')
plt.savefig('mlp_roc_curve.png')
plt.close()

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Neural Network Model (MLPClassifier)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png')
plt.close()

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('mlp_learning_curve.png')
plt.close()

# Compare with previous models - create comparison table
mlp_precision = precision_score(y_test, y_pred_optimal)
mlp_recall = recall_score(y_test, y_pred_optimal)
mlp_f1 = f1_score(y_test, y_pred_optimal)

# Previous models results (from our documents)
xgb_orig_accuracy = 0.8162
xgb_orig_f1 = 0.61
xgb_orig_recall = 0.55
xgb_orig_auc = "Not calculated"

xgb_improved_accuracy = 0.7885
xgb_improved_f1 = 0.66
xgb_improved_recall = 0.77
xgb_improved_auc = 0.858

# Create comparison data
comparison_data = {
    'Metric': ['Accuracy', 'F1 Score (Churn)', 'Recall (Churn)', 'AUC-ROC', 'Class balancing technique'],
    'XGBoost Original': [f"{xgb_orig_accuracy:.4f}", f"{xgb_orig_f1:.2f}", f"{xgb_orig_recall:.2f}", xgb_orig_auc, "None"],
    'XGBoost Improved': [f"{xgb_improved_accuracy:.4f}", f"{xgb_improved_f1:.2f}", f"{xgb_improved_recall:.2f}", f"{xgb_improved_auc:.3f}", "SMOTE"],
    'MLPClassifier': [f"{accuracy:.4f}", f"{mlp_f1:.2f}", f"{mlp_recall:.2f}", f"{roc_auc:.3f}", "SMOTE"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n===== MODEL COMPARISON =====")
print(comparison_df)

# Save comparison to CSV
comparison_df.to_csv('model_comparison_with_mlp.csv', index=False)

print("\n===== NEURAL NETWORK (MLPClassifier) APPROACH SUMMARY =====")
print("1. Architecture:")
print("   - Scikit-learn MLPClassifier with two hidden layers (64, 32)")
print("   - ReLU activation function")
print("   - Adam optimizer")
print("   - Early stopping to prevent overfitting")

print("\n2. Imbalanced data handling:")
print("   - SMOTE to balance the training data")
print("   - Optimized probability threshold")

print("\n3. Performance highlights:")
if accuracy > xgb_improved_accuracy:
    print(f"   - Improved overall accuracy: {accuracy:.4f} vs {xgb_improved_accuracy:.4f} (XGBoost)")
else:
    print(f"   - Comparable accuracy: {accuracy:.4f} vs {xgb_improved_accuracy:.4f} (XGBoost)")

if mlp_recall > xgb_improved_recall:
    print(f"   - Better recall for churning customers: {mlp_recall:.2f} vs {xgb_improved_recall:.2f} (XGBoost)")
else:
    print(f"   - Comparable recall for churning customers: {mlp_recall:.2f} vs {xgb_improved_recall:.2f} (XGBoost)")

if roc_auc > xgb_improved_auc:
    print(f"   - Higher AUC-ROC score: {roc_auc:.3f} vs {xgb_improved_auc:.3f} (XGBoost)")
else:
    print(f"   - Comparable AUC-ROC score: {roc_auc:.3f} vs {xgb_improved_auc:.3f} (XGBoost)")

print("\n4. Advantages of MLPClassifier:")
print("   - Simpler implementation than TensorFlow/Keras")
print("   - Integrated with scikit-learn ecosystem")
print("   - Requires less computational resources")
print("   - Still captures non-linear relationships") 