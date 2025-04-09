import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = xgb.XGBClassifier()
model.load_model('xgboost_churn_model.json')

print("===== USING THE MODEL FOR NEW PREDICTIONS =====")
print("Demonstrating how to use the trained model to predict churn for new customers")

# Function to preprocess new customer data
def preprocess_customer_data(customer_data):
    """Preprocess new customer data to match the training data format"""
    # Load a sample of the original dataset to get column names and categorical values
    original_data = pd.read_csv('telco-customer-churn.csv')
    
    # Drop customerID and Churn from original data
    original_X = original_data.drop(['customerID', 'Churn'], axis=1)
    
    # Convert TotalCharges to numeric
    original_X['TotalCharges'] = pd.to_numeric(original_X['TotalCharges'], errors='coerce')
    original_X['TotalCharges'] = original_X['TotalCharges'].fillna(0)
    
    # Identify categorical columns (excluding TotalCharges and MonthlyCharges)
    categorical_columns = original_X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    # Create and fit label encoders for each categorical column
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(original_data[column])
    
    # Apply the same encoding to new data
    for column in categorical_columns:
        if column in customer_data.columns:
            customer_data[column] = label_encoders[column].transform(customer_data[column])
    
    # Ensure all columns match the original dataset
    for col in original_X.columns:
        if col not in customer_data.columns:
            print(f"Warning: Column {col} is missing in new data. Adding with default values.")
            customer_data[col] = 0
    
    # Return customer data with columns in the same order as training data
    return customer_data[original_X.columns]

# Example: Create sample new customer data
print("\n--- Creating sample new customers ---")
new_customers = pd.DataFrame({
    'gender': ['Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': [0, 1, 0, 1],
    'Partner': ['Yes', 'No', 'Yes', 'No'],
    'Dependents': ['No', 'No', 'Yes', 'Yes'],
    'tenure': [6, 24, 48, 12],
    'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
    'MultipleLines': ['No', 'Yes', 'No phone service', 'No'],
    'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'Yes', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No', 'No internet service'],
    'TechSupport': ['No', 'No', 'Yes', 'No internet service'],
    'StreamingTV': ['Yes', 'Yes', 'No', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
    'PaperlessBilling': ['Yes', 'Yes', 'No', 'No'],
    'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)'],
    'MonthlyCharges': [70.35, 89.10, 45.50, 30.20],
    'TotalCharges': [420.90, 2138.40, 2183.90, 362.40]
})

# Display the sample customers
print("\nSample New Customers:")
print(new_customers[['tenure', 'Contract', 'MonthlyCharges', 'InternetService']])

# Preprocess the new customer data
preprocessed_data = preprocess_customer_data(new_customers.copy())

# Make predictions
churn_probability = model.predict_proba(preprocessed_data)[:, 1]
churn_prediction = model.predict(preprocessed_data)

# Add predictions to the original data
new_customers['Churn_Probability'] = churn_probability
new_customers['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in churn_prediction]

# Display predictions
print("\n--- Churn Predictions ---")
results = new_customers[['tenure', 'Contract', 'MonthlyCharges', 'InternetService', 'Churn_Probability', 'Predicted_Churn']]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results)

print("\n===== CUSTOMER RISK ASSESSMENT =====")

# Classify customers by risk level
new_customers['Risk_Level'] = pd.cut(
    new_customers['Churn_Probability'], 
    bins=[0, 0.3, 0.6, 1.0], 
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Display risk levels
risk_count = new_customers['Risk_Level'].value_counts()
print("\nCustomer Risk Levels:")
for risk, count in risk_count.items():
    print(f"{risk}: {count} customers")

print("\n--- Recommended Actions ---")
print("High Risk Customers: Immediate retention offers and personalized outreach")
print("Medium Risk Customers: Targeted satisfaction surveys and service reviews")
print("Low Risk Customers: Standard loyalty program enrollment")

# Business impact estimate
avg_customer_value = new_customers['MonthlyCharges'].mean() * 12  # Annual value
high_risk_customers = new_customers[new_customers['Risk_Level'] == 'High Risk']
potential_revenue_loss = len(high_risk_customers) * avg_customer_value

print(f"\nPotential Annual Revenue at Risk: ${potential_revenue_loss:.2f}")
print("Cost of Targeted Retention per High-Risk Customer: $50.00")
print(f"Total Retention Campaign Cost: ${len(high_risk_customers) * 50:.2f}")
print(f"ROI if 50% of High-Risk Customers are Retained: {(potential_revenue_loss * 0.5 - len(high_risk_customers) * 50) / (len(high_risk_customers) * 50):.2f}x") 