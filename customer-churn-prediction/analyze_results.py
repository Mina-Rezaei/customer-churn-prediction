import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("===== FEATURE IMPORTANCE ANALYSIS =====")
print("Analyzing which factors most influence customer churn...")

# Display original feature importance plot
plt.figure(figsize=(12, 6))
img = plt.imread('feature_importance.png')
plt.imshow(img)
plt.axis('off')
plt.title('Feature Importance from Initial Model')
plt.show()

# Display optimized model feature importance
plt.figure(figsize=(12, 6))
img = plt.imread('final_feature_importance.png')
plt.imshow(img)
plt.axis('off')
plt.title('Feature Importance from Optimized Model')
plt.show()

# Display confusion matrix
plt.figure(figsize=(10, 6))
img = plt.imread('confusion_matrix.png')
plt.imshow(img)
plt.axis('off')
plt.title('Confusion Matrix')
plt.show()

print("\n===== KEY INSIGHTS =====")
print("1. Top factors influencing customer churn (based on feature importance):")
print("   - Contract type: Month-to-month contracts have higher churn rates")
print("   - Tenure: Newer customers are more likely to churn")
print("   - Monthly charges: Higher charges correlate with higher churn")
print("   - Internet service type: Fiber optic users may have higher churn")
print("   - Payment method: Electronic checks may indicate higher churn risk")

print("\n2. Model Performance Analysis:")
print("   - Strong at identifying loyal customers (91% recall for non-churning)")
print("   - Less effective at identifying churning customers (55% recall)")
print("   - Overall accuracy: 81.62% with optimized parameters")

print("\n===== SUGGESTIONS FOR IMPROVEMENT =====")
print("1. Address class imbalance using techniques like:")
print("   - SMOTE: Synthetic Minority Over-sampling Technique")
print("   - Class weights: Penalize misclassifications of minority class more")
print("   - Adjust classification threshold: Lower the threshold for positive class")

print("\n2. Feature engineering opportunities:")
print("   - Create interaction features between correlated variables")
print("   - Bin continuous variables (e.g., tenure groups, charge brackets)")
print("   - Create customer segments based on usage patterns")

print("\n3. Try additional models:")
print("   - Ensemble models: Random Forest, Stacking")
print("   - Neural networks for complex patterns")
print("   - CatBoost for better handling of categorical features")

print("\n===== BUSINESS RECOMMENDATIONS =====")
print("1. Focus retention efforts on:")
print("   - New customers in the first year")
print("   - Customers with month-to-month contracts")
print("   - Customers with high monthly charges")
print("   - Customers using electronic check payment method")

print("\n2. Potential retention strategies:")
print("   - Offer incentives to switch to longer-term contracts")
print("   - Create loyalty programs for new customers")
print("   - Review pricing for services with high churn rates")
print("   - Improve service quality for fiber optic internet users") 