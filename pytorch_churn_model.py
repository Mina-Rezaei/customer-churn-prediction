import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create output directory if it doesn't exist
output_dir = 'pytorch_model_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset class
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)  # Changed to FloatTensor for BCEWithLogitsLoss
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural Network Architecture
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
        self.output = nn.Linear(32, 1)  # Single output for binary classification
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.dropout3(x)
        x = self.output(x)  # Output logits
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
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
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_pytorch_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.sigmoid(outputs)  # Convert logits to probabilities
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pytorch_learning_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, (y_pred > threshold).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'pytorch_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'pytorch_roc_curve.png'))
    plt.close()

def main():
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv('telco-customer-churn.csv')
    
    # Drop customerID column
    df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    
    # Convert Churn to binary
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Feature engineering
    print("Performing feature engineering...")
    
    # Create tenure groups and encode them
    df['TenureGroup'] = pd.qcut(df['tenure'], q=5, labels=False)  # Using numeric labels 0-4
    
    # Create monthly charge buckets and encode them
    df['MonthlyChargeGroup'] = pd.qcut(df['MonthlyCharges'], q=5, labels=False)  # Using numeric labels 0-4
    
    # Count number of services
    service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[service_columns].apply(lambda x: sum(x == 'Yes'), axis=1)
    
    # Create interaction features
    df['TenureCharges'] = df['tenure'] * df['MonthlyCharges']
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for handling imbalanced data
    print("Applying SMOTE for handling imbalanced data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Create data loaders
    train_dataset = ChurnDataset(X_train_balanced, y_train_balanced)
    test_dataset = ChurnDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ChurnNet(input_size=X_train.shape[1]).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print("Starting training...")
    n_epochs = 50
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, n_epochs, device)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_pytorch_model.pth')))
    _, test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (test_preds > threshold).astype(int)
        f1 = f1_score(test_labels, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Save optimal threshold
    with open(os.path.join(output_dir, 'pytorch_optimal_threshold.txt'), 'w') as f:
        f.write(f'Optimal threshold: {best_threshold:.3f}')
    
    # Generate predictions with optimal threshold
    y_pred = (test_preds > best_threshold).astype(int)
    
    # Calculate and save metrics
    accuracy = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred)
    auc_roc = roc_auc_score(test_labels, test_preds)
    
    with open(os.path.join(output_dir, 'pytorch_metrics.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n\n')
        f.write('Classification Report:\n')
        f.write(report)
        f.write(f'\nAUC-ROC Score: {auc_roc:.4f}')
    
    # Plot learning curves
    plot_learning_curves(train_losses, val_losses)
    
    # Plot confusion matrix and ROC curve
    plot_confusion_matrix(test_labels, test_preds, best_threshold)
    plot_roc_curve(test_labels, test_preds)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC Score: {auc_roc:.4f}")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    print("\nDetailed metrics saved to pytorch_metrics.txt")
    print("Visualizations saved to pytorch_*.png files")

if __name__ == "__main__":
    main() 