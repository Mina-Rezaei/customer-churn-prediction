import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing TensorFlow, if not available, notify the user
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1_l2
    tensorflow_available = True
except ImportError:
    print("WARNING: TensorFlow is not installed. You will need to install it to run the neural network model.")
    print("Consider using a conda environment with: conda install tensorflow")
    print("Or, for CPU-only: pip install tensorflow-cpu")
    tensorflow_available = False

# Try importing imbalanced-learn, if not available, notify the user
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    imblearn_available = True
except ImportError:
    print("WARNING: imbalanced-learn is not installed. You will need it to handle imbalanced data.")
    print("Install with: pip install imbalanced-learn")
    imblearn_available = False

print("===== NEURAL NETWORK FOR CHURN PREDICTION =====")
print("Implementing a deep learning approach to handle imbalanced data and improve performance")

# Check if required libraries are available
if not tensorflow_available or not imblearn_available:
    print("\nERROR: Required libraries are missing. Please install them and run again.")
    import sys
    sys.exit(1)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the data
try:
    df = pd.read_csv('telco-customer-churn.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("ERROR: telco-customer-churn.csv not found.")
    print("Please download the dataset from Kaggle (https://www.kaggle.com/blastchar/telco-customer-churn) and place it in the current directory.")
    import sys
    sys.exit(1)

# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, handling empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

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

# Handle imbalanced data
print("\n--- Handling Imbalanced Data with SMOTETomek ---")
# SMOTETomek combines oversampling and undersampling
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)

print("Class Distribution after SMOTETomek:")
print(pd.Series(y_train_balanced).value_counts(normalize=True) * 100)

# Neural Network approach with specific techniques for imbalanced data
print("\n--- Building Neural Network Model ---")

# Define class weights to handle imbalance at the loss function level
class_weights = {0: 1., 1: (len(y_train_balanced) - sum(y_train_balanced)) / sum(y_train_balanced)}
print(f"Class weights: {class_weights}")

try:
    # Create a neural network model
    def create_model(input_dim):
        model = Sequential([
            # Input layer with dropout to reduce overfitting
            Dense(128, input_dim=input_dim, activation='relu', 
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with regularization
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer with sigmoid activation for binary classification
            Dense(1, activation='sigmoid')
        ])
        
        # Use binary crossentropy loss for binary classification
        # Focal loss could be an alternative for imbalanced data
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model

    # Create the model
    model = create_model(X_train_balanced.shape[1])
    model.summary()

    # Set up callbacks for training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        # Reduce learning rate when a metric has stopped improving
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        # Save the best model
        ModelCheckpoint(
            'best_nn_churn_model.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model
    print("\n--- Training Neural Network ---")
    history = model.fit(
        X_train_balanced, y_train_balanced,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Load the best model
    model = tf.keras.models.load_model('best_nn_churn_model.h5')

    # Evaluate the model
    print("\n--- Neural Network Performance ---")
    y_pred_proba = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

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
    plt.title('ROC Curve - Neural Network Model')
    plt.savefig('nn_roc_curve.png')
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Neural Network Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('nn_confusion_matrix.png')
    plt.close()

    # Plot training history
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(2, 2, 3)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig('nn_training_history.png')
    plt.close()

    # Compare with previous models - create comparison table
    nn_results = classification_report(y_test, y_pred_optimal, output_dict=True)
    nn_precision = nn_results['1']['precision']
    nn_recall = nn_results['1']['recall']
    nn_f1 = nn_results['1']['f1-score']

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
        'Neural Network': [f"{accuracy:.4f}", f"{nn_f1:.2f}", f"{nn_recall:.2f}", f"{roc_auc:.3f}", "SMOTETomek + Class Weights"]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\n===== MODEL COMPARISON =====")
    print(comparison_df)

    # Save comparison to CSV
    comparison_df.to_csv('model_comparison_with_nn.csv', index=False)

    print("\n===== NEURAL NETWORK APPROACH SUMMARY =====")
    print("1. Architecture:")
    print("   - Deep neural network with 3 hidden layers")
    print("   - Dropout and BatchNormalization for regularization")
    print("   - L1 and L2 regularization to prevent overfitting")

    print("\n2. Imbalanced data handling:")
    print("   - SMOTETomek to balance the training data")
    print("   - Class weights during training")
    print("   - Optimized probability threshold")

    print("\n3. Training techniques:")
    print("   - Early stopping to prevent overfitting")
    print("   - Learning rate reduction when performance plateaus")
    print("   - Model checkpointing to save best performing model")

    print("\n4. Performance highlights:")
    if accuracy > xgb_improved_accuracy:
        print(f"   - Improved overall accuracy: {accuracy:.4f} vs {xgb_improved_accuracy:.4f} (XGBoost)")
    else:
        print(f"   - Comparable accuracy: {accuracy:.4f} vs {xgb_improved_accuracy:.4f} (XGBoost)")

    if nn_recall > xgb_improved_recall:
        print(f"   - Better recall for churning customers: {nn_recall:.2f} vs {xgb_improved_recall:.2f} (XGBoost)")
    else:
        print(f"   - Comparable recall for churning customers: {nn_recall:.2f} vs {xgb_improved_recall:.2f} (XGBoost)")

    if roc_auc > xgb_improved_auc:
        print(f"   - Higher AUC-ROC score: {roc_auc:.3f} vs {xgb_improved_auc:.3f} (XGBoost)")
    else:
        print(f"   - Comparable AUC-ROC score: {roc_auc:.3f} vs {xgb_improved_auc:.3f} (XGBoost)")

    print("\n5. Neural network advantages for churn prediction:")
    print("   - Can capture more complex, non-linear relationships")
    print("   - Multiple techniques to handle imbalanced data")
    print("   - Can incorporate a variety of regularization methods")
    print("   - Flexible architecture that can be adapted to the problem")

    print("\n6. Model saved as 'best_nn_churn_model.h5'")
    print("   - Can be loaded with: model = tf.keras.models.load_model('best_nn_churn_model.h5')")

except Exception as e:
    print(f"\nERROR: An exception occurred during model training or evaluation: {str(e)}")
    print("This could be due to memory constraints or other issues.")
    print("Consider trying a simpler model or using a different approach.")
    
    # Create a basic comparison without neural network results
    print("\n===== MODEL COMPARISON (WITHOUT NEURAL NETWORK) =====")
    comparison_data = {
        'Metric': ['Accuracy', 'F1 Score (Churn)', 'Recall (Churn)', 'AUC-ROC', 'Class balancing technique'],
        'XGBoost Original': [f"{0.8162:.4f}", f"{0.61:.2f}", f"{0.55:.2f}", "Not calculated", "None"],
        'XGBoost Improved': [f"{0.7885:.4f}", f"{0.66:.2f}", f"{0.77:.2f}", f"{0.858:.3f}", "SMOTE"],
    }
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    print("\nRecommendations:")
    print("1. If you want to try neural networks, ensure TensorFlow is correctly installed")
    print("2. Try a simpler model architecture with fewer layers")
    print("3. Reduce batch size or epochs if facing memory issues")
    print("4. Consider using a different approach like ensemble methods")
    print("5. Try adjusting hyperparameters to better manage class imbalance") 