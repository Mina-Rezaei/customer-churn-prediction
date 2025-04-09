import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score

print("===== COMPARING ORIGINAL VS IMPROVED CHURN PREDICTION MODELS =====")

# Load the original and improved feature importance plots
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
img1 = plt.imread('final_feature_importance.png')
plt.imshow(img1)
plt.axis('off')
plt.title('Original Model Feature Importance', fontsize=14)

plt.subplot(1, 2, 2)
img2 = plt.imread('improved_feature_importance.png')
plt.imshow(img2)
plt.axis('off')
plt.title('Improved Model Feature Importance', fontsize=14)

plt.tight_layout()
plt.savefig('model_comparison_feature_importance.png')
plt.close()

# Load confusion matrices
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
img3 = plt.imread('confusion_matrix.png')
plt.imshow(img3)
plt.axis('off')
plt.title('Original Model Confusion Matrix', fontsize=14)

plt.subplot(1, 2, 2)
img4 = plt.imread('improved_confusion_matrix.png')
plt.imshow(img4)
plt.axis('off')
plt.title('Improved Model Confusion Matrix', fontsize=14)

plt.tight_layout()
plt.savefig('model_comparison_confusion_matrix.png')
plt.close()

# Load ROC curve for improved model
plt.figure(figsize=(10, 6))
img5 = plt.imread('improved_roc_curve.png')
plt.imshow(img5)
plt.axis('off')
plt.title('ROC Curve for Improved Model', fontsize=14)
plt.savefig('roc_curve_improved.png')
plt.close()

# Create comparison table
data = {
    'Metric': ['Accuracy', 'Class imbalance handling', 'F1 score (Churned customers)', 
               'Recall (Churned customers)', 'Feature engineering', 'AUC-ROC'],
    'Original Model': ['81.62%', 'None', '0.61', '0.55', 'Basic', 'Not calculated'],
    'Improved Model': ['79.79%', 'SMOTE', '0.66', '0.77', 'Advanced', '0.858']
}

comparison_df = pd.DataFrame(data)
print("\n===== MODEL COMPARISON =====")
print(comparison_df)

# Save comparison table as CSV
comparison_df.to_csv('model_comparison.csv', index=False)

print("\n===== KEY IMPROVEMENTS =====")
print("1. Recall for churned customers increased from 55% to 77%")
print("   - This means we're now catching 77% of customers who will churn vs 55% before")
print("   - This is crucial for a churn model where missing churning customers is costly")

print("\n2. F1 score for churned customers improved from 0.61 to 0.66")
print("   - Better balance between precision and recall")

print("\n3. New features provide better insights for business action:")
print("   - Tenure groups show exactly when customers are most likely to churn")
print("   - Service count reveals relationship between service adoption and churn")
print("   - Contract-charge interaction reveals price sensitivity by contract type")

print("\n4. Class imbalance addressed through SMOTE")
print("   - Model now gives more importance to the minority class (churning customers)")
print("   - Optimized probability threshold further improves identification of churning customers")

print("\n5. Business value:")
print("   - Despite slightly lower overall accuracy (81.6% vs 79.8%), the improved model")
print("     is much better at identifying customers at risk of churning")
print("   - This translates to identifying 22% more at-risk customers")
print("   - For a company with 100,000 customers and 10% annual churn, this could mean")
print("     identifying 2,200 more at-risk customers for retention campaigns")

print("\n===== RECOMMENDED MODEL =====")
print("The improved model should be used for customer churn prediction due to its:")
print("1. Superior ability to identify customers likely to churn (higher recall)")
print("2. Better balance between precision and recall (F1 score)")
print("3. More sophisticated features that provide actionable business insights")
print("4. Optimal probability threshold specific to this business problem")

# Save this information to file
with open('model_comparison_report.txt', 'w') as f:
    f.write("===== CHURN MODEL COMPARISON REPORT =====\n\n")
    f.write("Original Model (XGBoost basic):\n")
    f.write("- Accuracy: 81.62%\n")
    f.write("- F1 score for churned customers: 0.61\n")
    f.write("- Recall for churned customers: 0.55\n")
    f.write("- No class imbalance handling\n")
    f.write("- Basic feature engineering\n\n")
    
    f.write("Improved Model (XGBoost advanced):\n")
    f.write("- Accuracy: 79.79%\n")
    f.write("- F1 score for churned customers: 0.66\n")
    f.write("- Recall for churned customers: 0.77\n")
    f.write("- SMOTE used for handling class imbalance\n")
    f.write("- Advanced feature engineering\n")
    f.write("- AUC-ROC: 0.858\n")
    f.write("- Optimal probability threshold: 0.4506\n\n")
    
    f.write("Business Impact:\n")
    f.write("The improved model identifies 22% more customers at risk of churning.\n")
    f.write("For a company with 100,000 customers and 10% annual churn rate,\n")
    f.write("this means identifying 2,200 more at-risk customers for retention campaigns.\n\n")
    
    f.write("Recommendation: Use the improved model for production deployment.\n")

print("\nComparison report and visualizations saved to files.") 