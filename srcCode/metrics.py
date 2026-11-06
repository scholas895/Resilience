import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# ------------------------------
# 1. Load dataset
# ------------------------------
file_path = 'wustl-ehms-2020_with_attacks_categories.csv'  # adjust path if needed
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Attack categories: {df['Attack Category'].unique()}")

# ------------------------------
# 2. Basic cleaning
# ------------------------------
# Drop rows with missing vital signs (if any)
vital_cols = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
df_clean = df.dropna(subset=vital_cols).copy()

# Convert 'Attack Category' to categorical
df_clean['Attack Category'] = df_clean['Attack Category'].astype('category')

# For efficiency, sample a fraction (e.g., 10%) of the data for plotting
df_sample = df_clean.sample(frac=0.1, random_state=42)
print(f"Sampled shape: {df_sample.shape}")

# ------------------------------
# 3. Time-series plot of SpO2 under normal vs data alteration
# ------------------------------
# Select a contiguous block of normal and attack records for visualization
normal_block = df_sample[df_sample['Attack Category'] == 'normal'].iloc[:200]
attack_block = df_sample[df_sample['Attack Category'] == 'Data Alteration'].iloc[:200]

plt.figure(figsize=(12, 5))
plt.plot(normal_block.index, normal_block['SpO2'], label='Normal', color='blue', alpha=0.7)
plt.plot(attack_block.index, attack_block['SpO2'], label='Data Alteration', color='red', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('SpO2 (%)')
plt.title('Impact of Data Alteration Attack on Oxygen Saturation (SpO2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spO2_attack_comparison.png', dpi=300)
plt.show()

# ------------------------------
# 4. Boxplot of network features by attack category
# ------------------------------
# Select a few network features for visualization
net_features = ['Load', 'Rate', 'TotPkts', 'Loss']
# Create a boxplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, feat in enumerate(net_features):
    # Remove outliers for better visibility (use 99th percentile cap)
    data = df_sample.copy()
    cap = data[feat].quantile(0.99)
    data[feat] = data[feat].clip(upper=cap)
    sns.boxplot(data=data, x='Attack Category', y=feat, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Distribution of {feat} by Attack Category')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('network_features_boxplot.png', dpi=300)
plt.show()

# ------------------------------
# 5. Prepare data for classification
# ------------------------------
# Select features: combine vital signs and network features
feature_cols = vital_cols + ['Load', 'Rate', 'TotPkts', 'Loss', 'SrcBytes', 'DstBytes']
# Ensure all selected columns exist (some might be missing in sampled data)
feature_cols = [col for col in feature_cols if col in df_clean.columns]

X = df_clean[feature_cols].fillna(0)  # simple imputation for missing
y = (df_clean['Attack Category'] != 'normal').astype(int)  # binary: 0 normal, 1 attack

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ------------------------------
# 6. Train a Random Forest classifier
# ------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict probabilities for ROC curve
y_score = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# ------------------------------
# 7. Plot ROC curve
# ------------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Attack Detection')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=300)
plt.show()

# ------------------------------
# 8. Confusion matrix
# ------------------------------
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Attack Detection')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# ------------------------------
# 9. Feature importance (optional)
# ------------------------------
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # top 10
plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances')
plt.barh(range(10), importances[indices], align='center')
plt.yticks(range(10), [feature_cols[i] for i in indices])
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

sns.boxplot(data=data, x='Attack Category', y=feat, hue='Attack Category', palette='Set2', legend=False, ax=axes[i])