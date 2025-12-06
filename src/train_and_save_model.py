# train_and_save_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# تعديل المسار لو لزم
ob_path = "../digital-health-obesity-prediction/data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

print("Loading", ob_path)
df = pd.read_csv(ob_path)

# حساب BMI إذا غير موجود
if "BMI" not in df.columns and "Height" in df.columns and "Weight" in df.columns:
    def bmi_row(r):
        try:
            h = float(r['Height'])
            w = float(r['Weight'])
            if h > 10: h = h/100.0
            return round(w/(h*h),2)
        except:
            return np.nan
    df['BMI'] = df.apply(bmi_row, axis=1)

# هدف متعدد الفئات كما في dataset
y_multi = df['NObeyesdad'].astype(str).copy()
# تعريف ثنائي: non-obese vs obese
non_obese_labels = [c for c in sorted(y_multi.unique()) if 'normal' in c.lower() or 'insufficient' in c.lower()]
y_bin = y_multi.apply(lambda x: 0 if x in non_obese_labels else 1).astype(int)

# اعرف الخصائص التي سنستخدمها (استبعد الهدف وعمود BMI)
exclude = {'NObeyesdad','BMI','id','Id','index'}
feature_cols = [c for c in df.columns if c not in exclude]

# تبويب رقمي وكتيجوري
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

# ملء القيم المفقودة
X = df[feature_cols].copy()
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("missing")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.2, random_state=42, stratify=y_multi)

# Preprocessor + pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
], remainder='drop')

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])

print("Training pipeline...")
pipeline.fit(X_train, y_train)

# Save wrapper with features and labels
out = {
    "model": pipeline,
    "features": feature_cols,
    "multiclass_labels": pipeline.named_steps['clf'].classes_.tolist(),
    "non_obese_labels": non_obese_labels
}

out_path = "app\\model.pkl"
joblib.dump(out, out_path)
print("Saved model to", out_path)
