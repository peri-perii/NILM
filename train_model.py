"""
train_model.py
--------------
Trains a Random Forest classifier that labels 5-minute power windows
with the dominant appliance (or "Standby" when only the fridge is running).

Feature engineering (per window of 5 samples):
  mean, std, max, min, range, median, q75, q25, slope, energy

Only "clean" windows — where a single non-fridge appliance dominates
more than 70 % of the window — are used for training, ensuring high
label quality and a realistic ≥ 80 % accuracy target.

Outputs saved by joblib:
  nilm_model.joblib   — trained RandomForestClassifier
  nilm_scaler.joblib  — fitted StandardScaler
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 5          # minutes per feature window
STEP_SIZE     = 1          # stride (minutes); overlapping windows for more data
PURITY_THRESH = 0.70       # fraction of window that must be dominated by one label
SEED          = 42

APPLIANCE_WATTAGES = {
    "Refrigerator":    150,
    "AC":              2000,
    "WashingMachine":  500,
    "Microwave":       1200,
    "TV":              100,
    "WaterHeater":     1500,
}

LABEL_MAP = {name: idx for idx, name in enumerate(APPLIANCE_WATTAGES.keys())}
LABEL_MAP["Standby"] = len(LABEL_MAP)
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

print("=" * 55)
print("  NILM Model Training -- Beyond the One Number")
print("=" * 55)


# ── 1. Load data ──────────────────────────────────────────────────────────────
try:
    df_power = pd.read_csv("household_power.csv", parse_dates=["timestamp"])
    df_truth = pd.read_csv("appliance_ground_truth.csv", parse_dates=["timestamp"])
except FileNotFoundError as exc:
    raise SystemExit(
        f"[ERROR] Missing CSV: {exc}. "
        "Run `python generate_data.py` first."
    )

df = df_power.merge(df_truth, on="timestamp")
total_rows = len(df)
print(f"  Loaded {total_rows} rows from household_power.csv")


# ── 2. Feature extraction ─────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract 10 statistical features from a 1-D power window."""
    mn    = np.mean(window)
    sd    = np.std(window)
    mx    = np.max(window)
    mi    = np.min(window)
    rng   = mx - mi
    med   = np.median(window)
    q75   = np.percentile(window, 75)
    q25   = np.percentile(window, 25)
    # slope — linear regression coefficient (normalised by window index)
    x     = np.arange(len(window), dtype=float)
    slope = np.polyfit(x, window, 1)[0]
    energy = np.sum(window) / 60 / 1000  # kWh in window
    return np.array([mn, sd, mx, mi, rng, med, q75, q25, slope, energy])


power_vals  = df["total_power_watts"].values
appliance_cols = list(APPLIANCE_WATTAGES.keys())
truth_vals  = df[appliance_cols].values   # shape (1440, 6)

X_list, y_list = [], []

for start in range(0, total_rows - WINDOW_SIZE + 1, STEP_SIZE):
    end    = start + WINDOW_SIZE
    window = power_vals[start:end]
    labels_in_window = truth_vals[start:end]   # (5, 6)

    # Determine dominant appliance in this window
    # A minute is "dominated" by the appliance with highest contribution
    minute_labels = []
    for row in labels_in_window:
        dominant_idx = np.argmax(row)
        dominant_val = row[dominant_idx]
        if dominant_val < 50:           # nothing significant active
            minute_labels.append("Standby")
        else:
            minute_labels.append(appliance_cols[dominant_idx])

    # Check window purity
    from collections import Counter
    counts  = Counter(minute_labels)
    top_lbl, top_cnt = counts.most_common(1)[0]

    if top_cnt / WINDOW_SIZE >= PURITY_THRESH:
        feats = extract_features(window)
        X_list.append(feats)
        y_list.append(LABEL_MAP[top_lbl])

X = np.array(X_list)
y = np.array(y_list)
print(f"  Windows extracted  : {len(X):,}")
print(f"  Label distribution :")
from collections import Counter
for lbl, cnt in sorted(Counter(y.tolist()).items()):
    print(f"    [{lbl}] {IDX_TO_LABEL[lbl]:<18} {cnt:>5} windows")


# ── 3. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ── 4. Scale features ─────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Train Random Forest ────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=SEED,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Test Accuracy : {accuracy * 100:.2f}%")
if accuracy >= 0.80:
    print("  [OK] Accuracy target (80%) achieved!")
else:
    print("  [!!] Accuracy below 80% -- consider adjusting purity threshold or features.")

target_names = [IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL)]
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ── 7. Feature importance ─────────────────────────────────────────────────────
feat_names = ["mean", "std", "max", "min", "range", "median", "q75", "q25", "slope", "energy"]
importances = model.feature_importances_
print("  Feature Importances:")
for fname, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    bar = "|" * int(imp * 40)
    print(f"    {fname:<8} {imp:.4f}  {bar}")

# ── 8. Save artifacts ─────────────────────────────────────────────────────────
import json

# Extract RandomForest parameters
n_classes = int(model.n_classes_)
classes = model.classes_.tolist()
estimators = model.estimators_

trees_data = []
for est in estimators:
    tree = est.tree_
    trees_data.append({
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value.squeeze(axis=1).tolist()  # shape (n_nodes, n_classes)
    })

# Extract StandardScaler parameters
scaler_data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "var": scaler.var_.tolist()
}

# Combine into a single JSON model
json_model = {
    "n_classes": n_classes,
    "classes": classes,
    "scaler": scaler_data,
    "trees": trees_data
}

with open("nilm_model.json", "w") as f:
    json.dump(json_model, f)

# Save label map for app.py
with open("label_map.json", "w") as f:
    json.dump({"label_map": LABEL_MAP, "idx_to_label": {str(k): v for k, v in IDX_TO_LABEL.items()}}, f, indent=2)

print("\n  Saved:")
print("    • nilm_model.json (Secure JSON-based Random Forest)")
print("    • label_map.json")
print("=" * 55)

