import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

print("=" * 50)
print("  DISASTER PREDICTION MODEL - TRAINING")
print("=" * 50)

# ── STEP 1: Load & Filter ──────────────────────────
print("\n[1/5] Data load ho raha hai...")
df = pd.read_csv('data/disasterIND.csv')

# Sirf top 4 disasters rakho (baaki ke paas data bahut kam hai)
df = df[df['Disaster Type'].isin(['Flood', 'Storm', 'Earthquake', 'Drought'])]
print(f"      Records after filtering: {len(df)}")
print(f"      Class distribution:\n{df['Disaster Type'].value_counts()}")

# ── STEP 2: Clean Data ─────────────────────────────
print("\n[2/5] Data clean ho raha hai...")
df['Total Deaths']   = pd.to_numeric(df['Total Deaths'],   errors='coerce').fillna(0)
df['Total Affected'] = pd.to_numeric(df['Total Affected'], errors='coerce').fillna(0)
df['Start Month']    = pd.to_numeric(df['Start Month'],    errors='coerce').fillna(6)
df['Start Year']     = pd.to_numeric(df['Start Year'],     errors='coerce').fillna(2000)

# ── STEP 3: Better Features ────────────────────────
print("\n[3/5] Better features ban rahe hain...")

# Season feature — month se season determine karo
# Monsoon (Jun-Sep)  → Flood likely
# Post-monsoon (Oct-Dec) → Cyclone/Storm likely
# Summer (Mar-May)   → Drought/Heatwave likely
# Winter (Jan-Feb)   → Earthquake/Cold wave likely
def get_season(month):
    if month in [6, 7, 8, 9]:    return 1  # Monsoon
    elif month in [10, 11, 12]:  return 2  # Post-monsoon
    elif month in [3, 4, 5]:     return 3  # Summer
    else:                         return 4  # Winter

df['Season']      = df['Start Month'].apply(get_season)

# Log transform — deaths/affected mein bahut bada range hai (1 se 20 lakh)
# Log se model zyada stable rehta hai
df['log_deaths']   = np.log1p(df['Total Deaths'])
df['log_affected'] = np.log1p(df['Total Affected'])

# Final features
FEATURES = ['Start Year', 'Start Month', 'Season', 'log_deaths', 'log_affected']
X = df[FEATURES]
y = df['Disaster Type']

# Train/Test split — stratify se har class dono mein jaegi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 4: SMOTE — Balance Classes ───────────────
print("\n[4/5] SMOTE se data balance ho raha hai...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"      After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

# ── STEP 5: Train Model ────────────────────────────
print("\n[5/5] Model train ho raha hai...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    max_depth=12
)
model.fit(X_res, y_res)

# ── RESULTS ───────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n{'='*50}")
print(f"  ACCURACY: {acc:.2f}%")
print(f"{'='*50}")
print(classification_report(y_test, y_pred))

# Feature importance
print("Feature Importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {feat:<15} {bar} {imp:.3f}")

# ── SAVE MODEL ────────────────────────────────────
# Model ke saath features bhi save karo (app.py mein kaam aayega)
save_data = {
    'model': model,
    'features': FEATURES
}
with open('model/disaster_model.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print(f"\n✅ Model save ho gaya! → model/disaster_model.pkl")
print("   Ab app.py update karo (neeche diya hua code dekho)\n")