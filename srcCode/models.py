import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')

# ------------------------------------------------------------
# 2. Prepare features and target
# ------------------------------------------------------------
# The target column is 'Label' (0 = normal, 1 = attack)
# Drop non‑numeric columns (they contain strings, IP addresses, MACs, etc.)
# Also drop 'Attack Category' because it's redundant with 'Label'.
non_numeric = ['Dir', 'Flgs', 'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac',
               'Attack Category', 'Label']
feature_cols = [c for c in df.columns if c not in non_numeric]

# Keep only rows where all feature values are finite (no missing)
X = df[feature_cols].select_dtypes(include=[np.number]).dropna()
y = df.loc[X.index, 'Label']   # align target with remaining rows

print(f"Using {X.shape[0]} samples and {X.shape[1]} features.")

# ------------------------------------------------------------
# 3. Split into train / test sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 4. Scale features (important for SVM, Logistic Regression)
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 5. Define models to compare
# ------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# ------------------------------------------------------------
# 6. Train each model and collect metrics
# ------------------------------------------------------------
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    })

# Convert to DataFrame for easy plotting
results_df = pd.DataFrame(results)
print(results_df)

# ------------------------------------------------------------
# 7. Create grouped bar chart
# ------------------------------------------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(results_df['Model']))  # the label locations
width = 0.2  # width of each bar
multiplier = 0

fig, ax = plt.subplots(figsize=(12, 6))

for metric in metrics:
    offset = width * multiplier
    bars = ax.bar(x + offset, results_df[metric], width, label=metric)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    multiplier += 1

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('ML Model Performance Comparison (Binary Classification)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)  # centre the group of bars
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.legend(loc='lower right', ncol=4)
ax.set_ylim(0, 1.05)  # all metrics are between 0 and 1
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()