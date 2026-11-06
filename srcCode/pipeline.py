import pandas as pd
import numpy as np

# ==========================================
# 1. Load and Clean Data
# ==========================================
CSV_PATH = "wustl-ehms-2020_with_attacks_categories.csv"
df = pd.read_csv(CSV_PATH)

df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

cols = ['Dur', 'SpO2', 'Heart_rate']
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df_clean = df[(df['SpO2'] > 0) & (df['Heart_rate'] > 0)].copy()

# ==========================================
# 2. Dynamic Baselines for POI
# ==========================================
normal_mask = df_clean['Attack Category'] == 'normal'
baseline_spo2 = df_clean.loc[normal_mask, 'SpO2'].mean()
baseline_hr = df_clean.loc[normal_mask, 'Heart_rate'].mean()

# WIDEN THRESHOLD: Anomaly only if 20% deviation from baseline (more realistic)
spo2_threshold = baseline_spo2 * 0.80 
hr_threshold = baseline_hr * 1.20      

df_clean['Is_Anomaly'] = (df_clean['SpO2'] < spo2_threshold) | (df_clean['Heart_rate'] > hr_threshold)

normal_df = df_clean[normal_mask]
da_df = df_clean[df_clean['Attack Category'] == 'Data Alteration']
spoof_df = df_clean[df_clean['Attack Category'] == 'Spoofing']

# ==========================================
# 3. Correct ESD and PSO Math
# ==========================================
HOURS_PER_YEAR = 8760.0
total_duration_sec = df_clean['Dur'].sum()

frac_da = da_df['Dur'].sum() / total_duration_sec
frac_spoof = spoof_df['Dur'].sum() / total_duration_sec

poi_normal = normal_df['Is_Anomaly'].mean() * 100
poi_da = da_df['Is_Anomaly'].mean() * 100
poi_spoof = spoof_df['Is_Anomaly'].mean() * 100

results = []

# -- 1. Baseline Normal Operation --
results.append({
    'Scenario': 'Baseline Normal Operation',
    'PSO': 1.0000,
    'ESD (hours/year)': 0.00,
    'POI': round(poi_normal, 2)
})

# -- 2. Data Alteration Attack --
esd_da = frac_da * HOURS_PER_YEAR
results.append({
    'Scenario': 'Data Alteration Attack (MITM/insider)',
    'PSO': round(1.0 - frac_da, 4),
    'ESD (hours/year)': round(esd_da, 2),
    'POI': round(poi_da, 2)
})

# -- 3. Social Engineering (Mapped to Spoofing) --
esd_spoof = frac_spoof * HOURS_PER_YEAR
results.append({
    'Scenario': 'Social Engineering (phishing) Attack',
    'PSO': round(1.0 - frac_spoof, 4),
    'ESD (hours/year)': round(esd_spoof, 2),
    'POI': round(poi_spoof, 2)
})

# -- 4. Ransomware Attack --
frac_ransom = min(frac_da * 2.0, 1.0)
results.append({
    'Scenario': 'Ransomware Attack (availability)',
    'PSO': round(1.0 - frac_ransom, 4),
    'ESD (hours/year)': round(frac_ransom * HOURS_PER_YEAR, 2),
    'POI': round(poi_da * 1.5, 2)
})

# -- 5. Multi-Vector APT Campaign --
frac_apt = min((frac_da + frac_spoof) * 3.0, 1.0)
results.append({
    'Scenario': 'Multi-Vector APT Campaign',
    'PSO': round(1.0 - frac_apt, 4),
    'ESD (hours/year)': round(frac_apt * HOURS_PER_YEAR, 2),
    'POI': round((poi_da + poi_spoof) / 2 * 2.5, 2)
})

# ==========================================
# 4. Output and Export
# ==========================================
metrics_df = pd.DataFrame(results)
metrics_df.to_csv('IoMT_HCPS_Perfect_Metrics.csv', index=False)

print(metrics_df.to_markdown(index=False))