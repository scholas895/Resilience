import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
# Replace the filename with the actual path if needed.
df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')

# ------------------------------------------------------------
# 2. Count the occurrences of each attack category
# ------------------------------------------------------------
# The column 'Attack Category' contains values like 'normal',
# 'Spoofing', 'Data Alteration'. We also keep the 'Label' column
# if you need to separate normal vs attack, but here we just use
# the category names.
category_counts = df['Attack Category'].value_counts()

# Print the counts for verification
print("Attack Category distribution:")
print(category_counts)
print("\nPercentages:")
print(category_counts / category_counts.sum() * 100)

# ------------------------------------------------------------
# 3. Create a bar chart (similar to many "class distribution" figures)
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(category_counts.index, category_counts.values,
               color=['#2ecc71', '#e74c3c', '#f39c12'])

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(height)}', ha='center', va='bottom', fontsize=11)

plt.title('Attack Category Distribution', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('attack_distribution_bar.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 4. (Optional) Create a table similar to the provided example
# ------------------------------------------------------------
# Build a summary DataFrame with counts and percentages
summary = category_counts.reset_index()
summary.columns = ['Attack Category', 'Count']
summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)
summary['Display'] = summary['Count'].astype(str) + ' (' + summary['Percentage'].astype(str) + '%)'

# Plot the table as a matplotlib table
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis('off')  # hide axes

table_data = [['Attack Category', 'Count (%)']] + summary[['Attack Category', 'Display']].values.tolist()
table = ax.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.3, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

plt.title('Attack Category Distribution (Table)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('attack_distribution_table.png', dpi=150)
plt.show()