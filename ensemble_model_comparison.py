import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tensorflow import keras
import os

# Create output directory if it doesn't exist
os.makedirs('ensemble_outputs', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the data for all models"""
    # Load data
    df = pd.read_csv('telco-customer-churn.csv')
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Feature engineering
    df['TenureGroup'] = pd.qcut(df['tenure'], q=4, labels=['Very New', 'New', 'Medium', 'Long-term'])
    service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[service_columns].apply(lambda x: sum(x == 'Yes'), axis=1)
    df['MonthlyChargeBucket'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    df['TenureCharges'] = df['tenure'] * df['MonthlyCharges']
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_balanced, y_train_balanced)
    return model

def train_keras(X_train, y_train):
    """Train Keras neural network"""
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Create model
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    # Train model
    model.fit(
        X_train_balanced, y_train_balanced,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    return model

class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ChurnNet(nn.Module):
    def __init__(self, input_size):
        super(ChurnNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.dropout3(x)
        x = self.output(x)
        return x

def train_pytorch(X_train, y_train, input_size):
    """Train PyTorch neural network"""
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Create dataset and dataloader
    train_dataset = ChurnDataset(X_train_balanced, y_train_balanced)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChurnNet(input_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step(train_loss)
        
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save(model.state_dict(), 'ensemble_outputs/best_pytorch_model.pth')
    
    return model

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return their predictions"""
    predictions = {}
    
    # XGBoost predictions
    xgb_preds = models['xgboost'].predict_proba(X_test)[:, 1]
    predictions['xgboost'] = xgb_preds
    
    # Keras predictions
    keras_preds = models['keras'].predict(X_test).flatten()
    predictions['keras'] = keras_preds
    
    # PyTorch predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models['pytorch'].eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        pytorch_preds = torch.sigmoid(models['pytorch'](X_test_tensor)).cpu().numpy().flatten()
    predictions['pytorch'] = pytorch_preds
    
    return predictions

def ensemble_voting(predictions, threshold=0.5):
    """Perform ensemble voting"""
    # Convert probabilities to binary predictions
    binary_preds = {model: (preds > threshold).astype(int) for model, preds in predictions.items()}
    
    # Majority voting
    ensemble_preds = np.zeros(len(predictions['xgboost']))
    for model in binary_preds:
        ensemble_preds += binary_preds[model]
    
    # If at least 2 models agree, use their prediction
    ensemble_preds = (ensemble_preds >= 2).astype(int)
    
    return ensemble_preds

def plot_ensemble_results(predictions, y_test):
    """Plot comparison of model performances"""
    # ROC curves
    plt.figure(figsize=(10, 6))
    for model, preds in predictions.items():
        fpr, tpr, _ = plt.roc_curve(y_test, preds)
        auc = roc_auc_score(y_test, preds)
        plt.plot(fpr, tpr, label=f'{model} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.savefig('ensemble_outputs/roc_curves.png')
    plt.close()
    
    # Confusion matrices
    for model, preds in predictions.items():
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, (preds > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'ensemble_outputs/confusion_matrix_{model}.png')
        plt.close()

def save_model_comparison(predictions, y_test):
    """Save model comparison results"""
    results = {}
    for model, preds in predictions.items():
        results[model] = {
            'Accuracy': accuracy_score(y_test, (preds > 0.5).astype(int)),
            'AUC-ROC': roc_auc_score(y_test, preds),
            'Classification Report': classification_report(y_test, (preds > 0.5).astype(int))
        }
    
    # Save results to file
    with open('ensemble_outputs/model_comparison.txt', 'w') as f:
        for model, metrics in results.items():
            f.write(f"\n{model.upper()} Model Results:\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write(f"AUC-ROC: {metrics['AUC-ROC']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(metrics['Classification Report'])
            f.write("\n" + "="*50 + "\n")

def main():
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = load_and_preprocess_data()
    
    # Train models
    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    print("Training Keras model...")
    keras_model = train_keras(X_train_scaled, y_train)
    
    print("Training PyTorch model...")
    pytorch_model = train_pytorch(X_train_scaled, y_train, X_train.shape[1])
    
    models = {
        'xgboost': xgb_model,
        'keras': keras_model,
        'pytorch': pytorch_model
    }
    
    # Get predictions from all models
    predictions = evaluate_models(models, X_test_scaled, y_test)
    
    # Perform ensemble voting
    ensemble_preds = ensemble_voting(predictions)
    predictions['ensemble'] = ensemble_preds
    
    # Plot results
    plot_ensemble_results(predictions, y_test)
    
    # Save comparison results
    save_model_comparison(predictions, y_test)
    
    # Save predictions for all customers
    all_predictions = pd.DataFrame({
        'XGBoost_Probability': predictions['xgboost'],
        'Keras_Probability': predictions['keras'],
        'PyTorch_Probability': predictions['pytorch'],
        'Ensemble_Prediction': predictions['ensemble']
    })
    all_predictions.to_csv('ensemble_outputs/all_predictions.csv', index=False)
    
    print("Model comparison and predictions saved to ensemble_outputs/")

if __name__ == "__main__":
    main() 