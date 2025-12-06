# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Obesity Risk Predictor", layout="wide")
st.title("Obesity Risk Predictor")

# ---------------------------
# Load Model (supports wrapped dict) - tries model_latest first
# ---------------------------
def load_model():
    candidates = ["app/model_latest.pkl", "app/model.pkl"]
    for path in candidates:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict) and "model" in obj:
                    st.sidebar.success(f"Loaded wrapped model: {os.path.basename(path)}")
                    return obj["model"], obj.get("features", None)
                st.sidebar.success(f"Loaded model: {os.path.basename(path)}")
                return obj, None
            except Exception as e:
                st.sidebar.warning(f"Found model at `{path}` but failed to load: {e}")
    st.sidebar.info("No usable model found (will use BMI fallback).")
    return None, None

# ---------------------------
# Utilities
# ---------------------------
def bmi_from_height_weight(h, w):
    try:
        h = float(h)
        w = float(w)
        if h > 10:
            h = h / 100.0
        return round(w / (h * h), 2)
    except Exception:
        return np.nan

def bmi_category(bmi):
    try:
        b = float(bmi)
    except:
        return "Unknown"
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal weight"
    if b < 30:
        return "Overweight"
    if b < 35:
        return "Obesity Class I"
    if b < 40:
        return "Obesity Class II"
    return "Obesity Class III"

def normalize_single(user_input):
    mapping = {
        "CALC": {"no":"no","sometimes":"Sometimes","frequently":"Frequently"},
        "CAEC": {"no":"no","sometimes":"Sometimes","frequently":"Frequently","always":"Always"},
        "FAVC": {"no":"no","yes":"yes"},
        "SCC": {"no":"no","yes":"yes"},
        "SMOKE": {"no":"no","yes":"yes"},
        "family_history_with_overweight": {"no":"no","yes":"yes"},
        "Gender": {"female":"Female","male":"Male"},
        "MTRANS": {
            "automobile":"Automobile","walking":"Walking",
            "public_transportation":"Public_Transportation","public transportation":"Public_Transportation",
            "bike":"Bike","motorbike":"Motorbike","no_transport":"No_transport","notransport":"No_transport"
        }
    }
    out = user_input.copy()
    for k,v in out.items():
        if isinstance(v, str):
            out[k] = v.strip()
    for key, m in mapping.items():
        if key in out:
            val = str(out.get(key,"")).strip().lower()
            if val in m:
                out[key] = m[val]
    if "Height" in out:
        try:
            hh = float(out["Height"])
            out["Height"] = hh/100.0 if hh > 10 else hh
        except:
            pass
    return out

def normalize_batch(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    def map_col(col, mp):
        if col in df.columns:
            df[col] = df[col].fillna("").apply(lambda x: mp.get(str(x).strip().lower(), x))
    map_col("CALC", {"no":"no","sometimes":"Sometimes","frequently":"Frequently"})
    map_col("CAEC", {"no":"no","sometimes":"Sometimes","frequently":"Frequently","always":"Always"})
    map_col("FAVC", {"no":"no","yes":"yes"})
    map_col("SCC", {"no":"no","yes":"yes"})
    map_col("SMOKE", {"no":"no","yes":"yes"})
    map_col("family_history_with_overweight", {"no":"no","yes":"yes"})
    map_col("Gender", {"female":"Female","male":"Male"})
    map_col("MTRANS", {"automobile":"Automobile","walking":"Walking","public transport":"Public_Transportation","public_transportation":"Public_Transportation","bike":"Bike","motorbike":"Motorbike","no_transport":"No_transport","notransport":"No_transport"})
    if "Height" in df.columns:
        def _to_m(x):
            try:
                xv = float(x)
                return xv/100.0 if xv > 10 else xv
            except:
                return np.nan
        df["Height"] = df["Height"].apply(_to_m)
    for n in ["Age","Weight","FCVC","NCP","CH2O","FAF","TUE"]:
        if n in df.columns:
            df[n] = pd.to_numeric(df[n], errors="coerce")
    return df

# ---------------------------
# Prediction helpers
# ---------------------------
def compute_obesity_probs(pipe, X, threshold=0.5):
    if pipe is None:
        raise RuntimeError("Model pipeline is None")
    try:
        proba_all = pipe.predict_proba(X)
    except Exception:
        preds = pipe.predict(X)
        clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else None
        classes = list(clf.classes_) if clf is not None and hasattr(clf, "classes_") else list(np.unique(preds))
        proba_all = np.zeros((len(preds), len(classes)))
        for i,p in enumerate(preds):
            try:
                j = classes.index(p)
                proba_all[i,j] = 1.0
            except:
                pass
    clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else None
    classes = None
    if clf is not None and hasattr(clf, "classes_"):
        classes = list(clf.classes_)
    if not classes or len(classes) == 0:
        classes = [f"class_{i}" for i in range(proba_all.shape[1])]
    obs_idx = []
    for i,c in enumerate(classes):
        try:
            cs = str(c).lower()
            if ("obes" in cs) or ("overweight" in cs):
                obs_idx.append(i)
        except:
            pass
    if not obs_idx:
        if proba_all.shape[1] == 2:
            obs_idx = [1]
        else:
            obs_idx = [i for i,c in enumerate(classes) if "normal" not in str(c).lower()]
    if not obs_idx:
        obs_idx = [proba_all.shape[1]-1]
    obesity_prob = np.sum(proba_all[:, obs_idx], axis=1)
    top_idx = np.argmax(proba_all, axis=1).astype(int)
    top_names = [classes[i] if i < len(classes) else str(i) for i in top_idx]
    obesity_pred = (obesity_prob >= threshold).astype(int)
    df_all_probs = pd.DataFrame(proba_all, columns=[str(c) for c in classes])
    return obesity_prob, obesity_pred, top_idx, top_names, df_all_probs

# ---------------------------
# Load model + features
# ---------------------------
model, saved_features = load_model()
training_features = None
if saved_features is not None:
    try:
        training_features = list(saved_features)
    except:
        training_features = None
if training_features is None and model is not None:
    try:
        pre = model.named_steps.get("pre")
        num_cols = []
        cat_cols = []
        try:
            num_cols = pre.transformers_[0][2]
        except Exception:
            pass
        try:
            cat_cols = pre.transformers_[1][2] if len(pre.transformers_)>1 else []
        except Exception:
            pass
        training_features = list(num_cols) + list(cat_cols)
    except Exception:
        training_features = None
if training_features is None:
    training_features = [
        "Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP",
        "SCC","SMOKE","CH2O","family_history_with_overweight",
        "FAF","TUE","CAEC","MTRANS"
    ]

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
debug = st.sidebar.checkbox("Show debug", value=False)
threshold = st.sidebar.slider("Obesity probability threshold", 0.0, 1.0, 0.5, 0.01)
st.caption("Note: The probability reflects the model’s estimate based on the input data. Use it as an indicator, not a medical diagnosis.")

# ---------------------------
# Single prediction UI
# ---------------------------
st.header("Single prediction")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=5, max_value=120, value=30)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=165.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
with c2:
    ch2o = st.number_input("Daily water (CH2O)", min_value=0.0, max_value=10.0, value=2.0)
    faf = st.selectbox("Physical activity (FAF)", [0,1,2,3], index=1)
    tue = st.selectbox("Time using tech (TUE)", [0,1,2], index=1)
with c3:
    gender = st.selectbox("Gender", ["Female","Male"])
    smoke = st.selectbox("Smoker (SMOKE)", ["no","yes"])
    family_over = st.selectbox("Family history with overweight", ["no","yes"])

favc = st.selectbox("FAVC (high-calorie food)", ["no","yes"])
fcvc = st.selectbox("FCVC (veg consumption 1-3)", [1,2,3], index=1)
ncp = st.selectbox("NCP (main meals 1-3)", [1,2,3], index=2)
caec = st.selectbox("CAEC (snacking)", ["no","sometimes","frequently","always"], index=0)
calc = st.selectbox("CALC (alcohol)", ["no","sometimes","frequently"], index=0)
scc = st.selectbox("SCC (monitoring)", ["no","yes"], index=0)
mtrans = st.selectbox("MTRANS (transport)", ["Automobile","Walking","Public_Transportation","Bike","Motorbike","No_transport"], index=1)

if st.button("Predict (single)"):
    user = {
        "Age": age, "Height": height, "Weight": weight, "CH2O": ch2o,
        "FAF": faf, "TUE": tue, "Gender": gender, "SMOKE": smoke,
        "family_history_with_overweight": family_over, "FAVC": favc,
        "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "CALC": calc,
        "SCC": scc, "MTRANS": mtrans
    }
    user = normalize_single(user)
    bmi = bmi_from_height_weight(user.get("Height", np.nan), user.get("Weight", np.nan))
    st.write(f"Calculated BMI: {bmi} — {bmi_category(bmi)}")

    X = pd.DataFrame([user])
    for c in training_features:
        if c not in X.columns:
            X[c] = np.nan
    X = X[training_features]

    if model is None:
        st.warning("Model not loaded — using BMI-threshold fallback.")
        prob = 1.0 if (not np.isnan(bmi) and bmi >= 30.0) else 0.0
        pred = int(prob >= 0.5)
        st.write({"obesity_prob": float(prob), "obesity_pred": int(pred)})
    else:
        try:
            obesity_prob, obesity_pred, top_idx, top_names, df_probs = compute_obesity_probs(model, X, threshold=threshold)
            prob_val = float(obesity_prob[0]) if len(obesity_prob)>0 else 0.0
            pct = prob_val * 100.0
            st.metric(label="Obesity probability", value=f"{pct:.2f} %")
            st.write("Risk meter:")
            st.progress(min(max(int(pct), 0), 100))

            # debug only: show class probabilities if requested
            if debug:
                st.write("DEBUG - Prepared X (passed to pipeline):")
                st.dataframe(X.head(1))
                st.write("DEBUG - All class probabilities (transposed):")
                st.dataframe(df_probs.T)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------
# Batch CSV mode
# ---------------------------
st.header("Batch prediction (CSV)")
uploaded = st.file_uploader("Upload CSV (columns similar to training)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview uploaded data:")
    st.dataframe(df.head())

    if st.button("Predict (batch)"):
        df2 = normalize_batch(df.copy())
        for c in training_features:
            if c not in df2.columns:
                df2[c] = np.nan
        if "BMI" not in df2.columns and "Height" in df2.columns and "Weight" in df2.columns:
            df2["BMI"] = df2.apply(lambda r: bmi_from_height_weight(r["Height"], r["Weight"]), axis=1)

        if model is None:
            st.warning("Model not loaded — using BMI threshold fallback.")
            if "BMI" in df2.columns:
                df2["obesity_prob"] = df2["BMI"].apply(lambda x: 1.0 if (not pd.isna(x) and x >= 30.0) else 0.0)
                df2["obesity_pred"] = (df2["obesity_prob"] >= 0.5).astype(int)
            else:
                df2["obesity_prob"] = 0.0
                df2["obesity_pred"] = 0
            st.dataframe(df2.head())
        else:
            try:
                X_batch = df2[training_features].copy()
                obesity_prob, obesity_pred, top_idx, top_names, df_probs = compute_obesity_probs(model, X_batch, threshold=threshold)

                # do not add multiclass label column; keep numeric outputs only
                df["obesity_prob"] = np.round(obesity_prob, 6)
                df["obesity_pred"] = obesity_pred

                # add percentage column for friendly display
                df["obesity_prob_pct"] = (df["obesity_prob"] * 100).round(2).astype(str) + " %"

                display_cols = ["Age","Gender","Height","Weight","BMI","obesity_prob_pct","obesity_pred"]
                display_cols = [c for c in display_cols if c in df.columns]
                st.success("Predictions added to uploaded dataframe.")
                st.dataframe(df[display_cols].head(50))

                if debug:
                    st.write("DEBUG - First rows of X_batch passed to pipeline:")
                    st.dataframe(X_batch.head(5))
                    st.write("DEBUG - class probability columns (first 5 rows):")
                    st.dataframe(df_probs.head(5))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

st.write("---")
st.write("If results look wrong: enable 'Show debug', upload a small CSV (5 rows) and copy the debug output here so I can inspect.")
