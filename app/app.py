# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Obesity Risk Predictor", layout="centered")

st.title("Obesity Risk Predictor — Wearable & Lifestyle")
st.write(
    "Predict obesity risk using the trained model. "
    "You can enter values manually or upload a CSV with the same columns used in training."
)

# ---------------------------
# Utility functions
# ---------------------------
def load_model(path="app/model.pkl"):
    try:
        model = joblib.load(path)
        st.success(f"Loaded model from `{path}`")
        return model
    except Exception as e:
        st.warning(f"Could not load model at `{path}`. The app will use BMI-threshold fallback. ({e})")
        return None

def bmi_from_row(row):
    h = row.get("Height", np.nan)
    w = row.get("Weight", np.nan)
    if pd.isna(h) or pd.isna(w):
        return np.nan
    try:
        h = float(h)
        if h > 10:  # assume cm -> convert to meters
            h = h / 100.0
        b = float(w) / (h ** 2)
        return round(b, 2)
    except Exception:
        return np.nan

def prepare_input_df(user_inputs, feature_order):
    df = pd.DataFrame([user_inputs])
    for c in feature_order:
        if c not in df.columns:
            df[c] = np.nan
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str)
    return df[feature_order]

def normalize_categories(user_input):
    mapping = {
        "CALC": {"no": "no", "sometimes": "Sometimes", "frequently": "Frequently"},
        "CAEC": {"no": "no", "sometimes": "Sometimes", "frequently": "Frequently", "always": "Always"},
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
    for col,m in mapping.items():
        if col in user_input:
            val = str(user_input.get(col,"")).strip().lower()
            if val in m:
                user_input[col] = m[val]
    # Height to meters
    try:
        if "Height" in user_input:
            h = float(user_input["Height"])
            user_input["Height"] = h/100.0 if h>10 else h
    except Exception:
        pass
    return user_input

def normalize_categories_batch(df):
    df = df.copy()
    if "CALC" in df.columns:
        df["CALC"] = df["CALC"].astype(str).str.strip().str.lower().map({
            "no":"no","sometimes":"Sometimes","frequently":"Frequently"
        }).fillna(df["CALC"])
    if "CAEC" in df.columns:
        df["CAEC"] = df["CAEC"].astype(str).str.strip().str.lower().map({
            "no":"no","sometimes":"Sometimes","frequently":"Frequently","always":"Always"
        }).fillna(df["CAEC"])
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map({"female":"Female","male":"Male"}).fillna(df["Gender"])
    for col in ["FAVC","SCC","SMOKE","family_history_with_overweight"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map({"no":"no","yes":"yes"}).fillna(df[col])
    if "MTRANS" in df.columns:
        df["MTRANS"] = df["MTRANS"].astype(str).str.strip().str.lower().map({
            "automobile":"Automobile","walking":"Walking",
            "public_transportation":"Public_Transportation","public transport":"Public_Transportation",
            "bike":"Bike","motorbike":"Motorbike","no_transport":"No_transport","notransport":"No_transport"
        }).fillna(df["MTRANS"])
    if "Height" in df.columns:
        def _to_m(x):
            try:
                xv = float(x)
                return xv/100.0 if xv>10 else xv
            except:
                return x
        df["Height"] = df["Height"].apply(_to_m)
    return df

def predict_with_model(model_obj, X):
    try:
        proba_all = model_obj.predict_proba(X)
        try:
            clf = model_obj.named_steps.get('clf', None)
            classes = list(clf.classes_) if clf is not None else None
        except Exception:
            classes = None
        if classes is None:
            try:
                classes = list(getattr(model_obj, "classes_", []))
            except Exception:
                classes = None
        if classes is not None and 1 in classes:
            idx1 = classes.index(1)
        else:
            idx1 = 1 if proba_all.shape[1] > 1 else 0
        proba = proba_all[:, idx1]
        preds = (proba >= st.session_state.get("threshold", 0.5)).astype(int)
        return proba, preds
    except Exception:
        try:
            if isinstance(model_obj, tuple) and len(model_obj)==3:
                clf, imputer, scaler = model_obj
                X_proc = scaler.transform(imputer.transform(X))
                proba_all = clf.predict_proba(X_proc)
                classes = list(clf.classes_)
                idx1 = classes.index(1) if 1 in classes else (1 if proba_all.shape[1]>1 else 0)
                proba = proba_all[:, idx1]
                preds = (proba >= st.session_state.get("threshold", 0.5)).astype(int)
                return proba, preds
        except Exception as e2:
            raise RuntimeError(f"Prediction failed in fallback: {e2}")
    raise RuntimeError("predict_with_model: could not compute predictions")

# ---------------------------
# Load model and features
# ---------------------------
model = load_model("app/model.pkl")
training_features = None
if model is not None:
    try:
        pre = model.named_steps.get("pre", None)
        if pre is not None:
            num_cols = pre.transformers_[0][2]
            cat_cols = pre.transformers_[1][2] if len(pre.transformers_)>1 else []
            training_features = list(num_cols) + list(cat_cols)
    except Exception:
        training_features = None

if training_features is None:
    training_features = [
        "Age","Gender","Height","Weight",
        "CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O",
        "family_history_with_overweight","FAF","TUE","CAEC","MTRANS"
    ]

# ---------------------------
# Sidebar global controls
# ---------------------------
st.sidebar.header("Input options")
st.sidebar.markdown("Change inputs or upload a CSV for batch predictions.")

# threshold slider (global)
threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
st.session_state["threshold"] = threshold

input_mode = st.sidebar.radio("Choose input method", ("Manual input (single)", "Upload CSV (batch)"))

# ---------------------------
# Manual input UI
# ---------------------------
if input_mode == "Manual input (single)":
    st.sidebar.subheader("Enter values")
    age = st.sidebar.number_input("Age", min_value=5, max_value=120, value=30, key="Age")
    height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=165.0, step=0.1, key="Height")
    weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1, key="Weight")
    ch2o = st.sidebar.number_input("Daily water intake (CH2O, scale)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="CH2O")
    faf = st.sidebar.selectbox("Physical activity frequency (FAF)", options=[0,1,2,3], index=1, key="FAF")
    tue = st.sidebar.selectbox("Time using tech (TUE)", options=[0,1,2], index=1, key="TUE")

    gender = st.sidebar.selectbox("Gender", ["Female","Male"], index=0, key="Gender")
    smoke = st.sidebar.selectbox("Smoker (SMOKE)", ["no","yes"], index=0, key="SMOKE")
    family_over = st.sidebar.selectbox("Family history with overweight", ["no","yes"], index=0, key="family_history_with_overweight")
    favc = st.sidebar.selectbox("Frequent high-calorie food (FAVC)", ["no","yes"], index=0, key="FAVC")
    fcvc = st.sidebar.selectbox("Vegetable consumption (FCVC) [1-3]", [1,2,3], index=1, key="FCVC")
    ncp = st.sidebar.selectbox("Number of main meals (NCP)", [1,2,3], index=2, key="NCP")
    caec = st.sidebar.selectbox("Snacking between meals (CAEC)", ["no","sometimes","frequently","always"], index=0, key="CAEC")
    calc = st.sidebar.selectbox("Alcohol consumption (CALC)", ["no","sometimes","frequently"], index=0, key="CALC")
    scc = st.sidebar.selectbox("Calorie consumption monitoring (SCC)", ["no","yes"], index=0, key="SCC")
    mtrans = st.sidebar.selectbox("Transportation (MTRANS)", ["Automobile","Walking","Public_Transportation","Bike","Motorbike","No_transport"], index=1, key="MTRANS")

    user_input = {
        "Age": age, "Height": height, "Weight": weight, "CH2O": ch2o,
        "FAF": faf, "TUE": tue, "Gender": gender, "SMOKE": smoke,
        "family_history_with_overweight": family_over, "FAVC": favc,
        "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "CALC": calc,
        "SCC": scc, "MTRANS": mtrans
    }

    bmi_val = bmi_from_row({"Height": height, "Weight": weight})
    st.subheader("Calculated BMI")
    if not np.isnan(bmi_val):
        st.metric("BMI", f"{bmi_val}")
    else:
        st.write("BMI could not be calculated (missing or invalid height/weight).")

    if st.button("Predict (single)"):
        # normalize user input & prepare
        user_input = normalize_categories(user_input)
        X = prepare_input_df(user_input, training_features)

        if model is None:
            prob = 1.0 if (not np.isnan(bmi_val) and bmi_val >= 30) else 0.0
            pred = 1 if prob >= threshold else 0
            st.warning("No saved model loaded; using BMI threshold fallback.")
            st.write({"obesity_prob": float(prob), "obesity_pred": int(pred)})
        else:
            try:
                proba, preds = predict_with_model(model, X)
                proba_val = float(np.round(proba[0], 6))
                st.success(f"Predicted obesity probability: {proba_val:.6f}")
                st.write(f"Using threshold = {threshold:.2f}")
                if preds[0] == 1:
                    st.error(f"Predicted class: 1 (Obese)")
                else:
                    st.success(f"Predicted class: 0 (Not Obese)")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------
# CSV upload mode
# ---------------------------
else:
    st.sidebar.subheader("CSV upload")
    st.sidebar.markdown("Upload a CSV with columns matching model features (e.g., Age, Height, Weight, Gender, FAVC, ...).")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.subheader("Preview uploaded data")
        st.dataframe(df_upload.head())

        missing = [c for c in training_features if c not in df_upload.columns]
        if missing:
            st.warning(f"Uploaded file is missing columns: {missing}. You can still try, but results may be unreliable.")

        if "Height" in df_upload.columns and "Weight" in df_upload.columns:
            df_upload["BMI"] = df_upload.apply(lambda r: bmi_from_row({"Height": r["Height"], "Weight": r["Weight"]}), axis=1)

        if st.button("Predict (batch)"):
            # normalize batch and ensure columns
            X_batch = normalize_categories_batch(df_upload.copy())
            for c in training_features:
                if c not in X_batch.columns:
                    X_batch[c] = np.nan

            if model is None:
                df_upload["obesity_prob"] = df_upload["BMI"].apply(lambda x: 1.0 if x >= 30 else 0.0)
                df_upload["obesity_pred"] = df_upload["obesity_prob"].apply(lambda x: int(x >= threshold))
                st.warning("No saved model loaded; used BMI-threshold fallback.")
                st.dataframe(df_upload.head())
            else:
                try:
                    proba, preds = predict_with_model(model, X_batch[training_features])
                    df_upload["obesity_prob"] = np.round(proba, 6)
                    df_upload["obesity_pred"] = preds
                    st.success("Predictions added to the uploaded dataframe.")
                    # show friendly percent column
                    df_upload["obesity_prob_pct"] = (df_upload["obesity_prob"]*100).round(2).astype(str) + "%"

                    # columns to display
                    display_cols = ["Age","Gender","Height","Weight","BMI","obesity_prob","obesity_prob_pct","obesity_pred"]
                    display_cols = [c for c in display_cols if c in df_upload.columns]

                    # create subset and style it
                    subset = df_upload[display_cols].copy()

                    def highlight_obese(row):
                        return ['background-color: #ffdddd' if row.get('obesity_pred', 0)==1 else '' for _ in row]

                    try:
                        styled = subset.style.apply(highlight_obese, axis=1)
                        st.dataframe(styled)
                    except Exception:
                        st.dataframe(subset)

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

            # if labels present, show metrics
            if "obesity_label" in df_upload.columns:
                y_true = df_upload["obesity_label"].astype(int)
                y_prob = df_upload["obesity_prob"].astype(float)
                y_pred = (y_prob >= threshold).astype(int)
                st.write("### Metrics on uploaded data")
                st.write("Accuracy", accuracy_score(y_true, y_pred))
                st.write("Precision", precision_score(y_true, y_pred, zero_division=0))
                st.write("Recall", recall_score(y_true, y_pred, zero_division=0))
                st.write("F1", f1_score(y_true, y_pred, zero_division=0))
                try:
                    st.write("AUC", roc_auc_score(y_true, y_prob))
                except Exception:
                    pass
                cm = confusion_matrix(y_true, y_pred)
                st.write("Confusion matrix")
                st.write(cm)

            # download cleaned CSV
            df_download = df_upload.copy()
            df_download["obesity_prob"] = df_download["obesity_prob"].round(6)
            csv = df_download.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# ---------------------------
# Feature importance (if available)
# ---------------------------
st.markdown("---")
st.subheader("Model explanation / Feature importance")

if model is None:
    st.info("No model loaded: feature importance not available.")
else:
    try:
        if hasattr(model.named_steps['clf'], "coef_"):
            coef = model.named_steps['clf'].coef_[0]
            try:
                pre = model.named_steps['pre']
                num_cols = pre.transformers_[0][2]
                cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []
                if cat_cols:
                    ohe = pre.named_transformers_['cat'].named_steps['onehot']
                    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
                else:
                    ohe_cols = []
                feature_names = list(num_cols) + list(ohe_cols)
            except Exception:
                feature_names = training_features
            feat_table = pd.DataFrame({"feature": feature_names, "coef": coef})
            feat_table = feat_table.reindex(feat_table.coef.abs().sort_values(ascending=False).index).head(10)
            st.table(feat_table)
        elif hasattr(model.named_steps['clf'], "feature_importances_"):
            clf = model.named_steps['clf']
            try:
                pre = model.named_steps['pre']
                num_cols = pre.transformers_[0][2]
                cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []
                if cat_cols:
                    ohe = pre.named_transformers_['cat'].named_steps['onehot']
                    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
                else:
                    ohe_cols = []
                feature_names = list(num_cols) + list(ohe_cols)
            except Exception:
                feature_names = training_features
            imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
            st.table(imp)
        else:
            st.write("No explainability info available for this model object.")
    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

st.markdown("---")
st.write("Notes: For production quality, ensure the model pipeline saved in `app/model.pkl` contains the preprocessor and classifier (Pipeline).")

        