import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (adjust the file path as needed)
df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')

# Map the numeric label to attack names (0 = normal, 1 = attack)
# The dataset already has an 'Attack Category' column, so we'll use that directly.
# If needed, you can also group by 'Label'.

# Select numeric columns to compare (excluding identifiers and MAC addresses)
features = [
    'SrcBytes', 'DstBytes', 'SrcLoad', 'DstLoad',
    'SrcGap', 'DstGap', 'SIntPkt', 'DIntPkt',
    'SrcJitter', 'DstJitter', 'sMaxPktSz', 'dMaxPktSz',
    'sMinPktSz', 'dMinPktSz', 'Dur', 'Rate',
    'Temp', 'SpO2', 'Pulse_Rate', 'Heart_rate', 'Resp_Rate', 'ST'
]

# Group by Attack Category and compute mean
grouped = df.groupby('Attack Category')[features].mean()

# Plot bar charts for each feature
grouped.T.plot(kind='bar', figsize=(14, 6), width=0.8)
plt.title('Mean Values of Network & Physiological Parameters by Attack Category')
plt.ylabel('Mean Value')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attack Category')
plt.tight_layout()
plt.show()

# Optional: Bar chart of most common source ports per attack category
# Source ports are in the 'Sport' column. We'll count the top 10 ports overall,
# then show their frequency per attack type.
top_ports = df['Sport'].value_counts().head(10).index
port_counts = df.groupby(['Attack Category', 'Sport']).size().unstack(fill_value=0)
port_counts = port_counts[top_ports]  # keep only top 10 ports

port_counts.T.plot(kind='bar', figsize=(12, 6))
plt.title('Top 10 Source Ports Used by Attack Category')
plt.ylabel('Number of Flows')
plt.xlabel('Source Port')
plt.xticks(rotation=0)
plt.legend(title='Attack Category')
plt.tight_layout()
plt.show()

# If you need to export the mean values for further use:
grouped.to_csv('mean_values_per_attack.csv')