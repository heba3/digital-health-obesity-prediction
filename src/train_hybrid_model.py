# train_hybrid_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# paths
aug_path = Path("data/processed/obesity_fitbit_augmented.csv")
out_model = Path("app/model.pkl")

# load
df = pd.read_csv(aug_path)

# target selection (adjust name if different)
target_candidates = ["NObeyesdad","Obesity_Level","NObeyesdad_Labels"]
target = None
for t in target_candidates:
    if t in df.columns:
        target = t
        break
if target is None:
    raise ValueError("Target column not found. Update target_candidates list.")

y = df[target].astype(str)
X = df.drop(columns=[target])

# features selection
num_core = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE"]
fit_cols = [c for c in X.columns if c.startswith("fit_")]
num_cols = [c for c in num_core if c in X.columns] + fit_cols
cat_cols = [c for c in ["Gender","family_history_with_overweight","FAVC","SCC","SMOKE","CAEC","CALC","MTRANS"] if c in X.columns]

# pipeline
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
], remainder="drop")

pipe = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X[num_cols + cat_cols], y, test_size=0.2, stratify=y, random_state=42)

# fit
print("Training...")
pipe.fit(X_train, y_train)

# evaluate
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# save model (pipeline)
joblib.dump(pipe, out_model)
print("Saved model to", out_model)
