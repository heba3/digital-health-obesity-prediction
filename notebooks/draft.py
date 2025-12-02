# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
        m = joblib.load(path)
        st.success(f"Loaded model from `{path}`")
        return m
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
        if h > 10:  # assume cm
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
    # ensure categorical columns are strings to avoid encoder issues
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str)
    return df[feature_order]

def normalize_categories(user_input):
    """
    Map common lowercase inputs to the exact category strings used during training.
    Extend mapping if you notice other mismatches.
    """
    mapping = {
        "CALC": {
            "no": "no",
            "sometimes": "Sometimes",
            "frequently": "Frequently"
        },
        "CAEC": {
            "no": "no",
            "sometimes": "Sometimes",
            "frequently": "Frequently",
            "always": "Always"
        }
    }
    for col, m in mapping.items():
        val = str(user_input.get(col, "")).strip().lower()
        if val in m:
            user_input[col] = m[val]
    return user_input

def predict_with_model(model_obj, X):
    """
    Robust wrapper: returns (proba_vector_for_class1, preds)
    Accepts Pipeline (recommended) or legacy tuple (clf, imputer, scaler).
    """
    # Try pipeline / object with predict_proba
    try:
        proba_all = model_obj.predict_proba(X)  # (n_samples, n_classes)
        # try to get classifier classes_
        try:
            clf = model_obj.named_steps.get('clf', None)
            classes = list(clf.classes_) if clf is not None else None
        except Exception:
            classes = None
            clf = None
        # fallback to model_obj classes_ if available
        if classes is None:
            try:
                classes = list(getattr(model_obj, "classes_", []))
            except Exception:
                classes = None

        # find index for class '1' if present
        if classes is not None and 1 in classes:
            idx1 = classes.index(1)
        else:
            # typical fallback: column 1 if more than 1 class, else 0
            idx1 = 1 if proba_all.shape[1] > 1 else 0

        proba = proba_all[:, idx1]
        preds = (proba >= 0.5).astype(int)
        return proba, preds

    except Exception:
        # fallback: maybe model_obj is (clf, imputer, scaler)
        try:
            if isinstance(model_obj, tuple) and len(model_obj) == 3:
                clf, imputer, scaler = model_obj
                X_proc = scaler.transform(imputer.transform(X))
                proba_all = clf.predict_proba(X_proc)
                classes = list(clf.classes_)
                idx1 = classes.index(1) if 1 in classes else (1 if proba_all.shape[1] > 1 else 0)
                proba = proba_all[:, idx1]
                preds = (proba >= 0.5).astype(int)
                return proba, preds
        except Exception as e2:
            raise RuntimeError(f"Prediction failed in fallback path: {e2}")
    raise RuntimeError("predict_with_model: could not compute predictions")

# ---------------------------
# Load model and determine feature list
# ---------------------------
model = load_model("app/model.pkl")

training_features = None
if model is not None:
    try:
        pre = model.named_steps.get("pre", None)
        if pre is not None:
            # Best-effort extraction of feature names (numeric + categorical original names)
            num_cols = pre.transformers_[0][2]
            cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []
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
# Sidebar & inputs
# ---------------------------
st.sidebar.header("Input options")
input_mode = st.sidebar.radio("Choose input method", ("Manual input (single)", "Upload CSV (batch)"))

# Presets in sidebar for quick testing
st.sidebar.markdown("### Quick presets")
if st.sidebar.button("Preset: Obese example (160 cm, 95 kg)"):
    st.session_state['Age'] = 45
    st.session_state['Height'] = 160.0
    st.session_state['Weight'] = 95.0
    st.session_state['CALC'] = "Sometimes"
    st.session_state['FAVC'] = "yes"
    st.session_state['FCVC'] = 1
    st.session_state['NCP'] = 3
    st.session_state['SCC'] = "no"
    st.session_state['SMOKE'] = "no"
    st.session_state['CH2O'] = 2
    st.session_state['family_history_with_overweight'] = "yes"
    st.session_state['FAF'] = 0
    st.session_state['TUE'] = 2
    st.session_state['CAEC'] = "Frequently"
    st.session_state['MTRANS'] = "Automobile"

if st.sidebar.button("Preset: Healthy example (170 cm, 60 kg)"):
    st.session_state['Age'] = 28
    st.session_state['Height'] = 170.0
    st.session_state['Weight'] = 60.0
    st.session_state['CALC'] = "no"
    st.session_state['FAVC'] = "no"
    st.session_state['FCVC'] = 3
    st.session_state['NCP'] = 2
    st.session_state['SCC'] = "no"
    st.session_state['SMOKE'] = "no"
    st.session_state['CH2O'] = 2
    st.session_state['family_history_with_overweight'] = "no"
    st.session_state['FAF'] = 2
    st.session_state['TUE'] = 1
    st.session_state['CAEC'] = "no"
    st.session_state['MTRANS'] = "Walking"

# ---------------------------
# Manual input UI
# ---------------------------
if input_mode == "Manual input (single)":
    st.sidebar.subheader("Enter values")
    age = st.sidebar.number_input("Age", min_value=5, max_value=120, value=st.session_state.get('Age', 30), key="Age")
    height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0,
                                     value=st.session_state.get('Height', 165.0), step=0.1, key="Height")
    weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0,
                                     value=st.session_state.get('Weight', 70.0), step=0.1, key="Weight")
    ch2o = st.sidebar.number_input("Daily water intake (CH2O, scale)", min_value=0.0, max_value=10.0,
                                   value=st.session_state.get('CH2O', 2.0), step=0.1, key="CH2O")
    faf = st.sidebar.selectbox("Physical activity frequency (FAF)", options=[0,1,2,3],
                               index=st.session_state.get('FAF', 1), key="FAF")
    tue = st.sidebar.selectbox("Time using tech (TUE)", options=[0,1,2],
                               index=st.session_state.get('TUE', 1), key="TUE")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0, key="Gender")
    smoke = st.sidebar.selectbox("Smoker (SMOKE)", ["no", "yes"], index=0, key="SMOKE")
    family_over = st.sidebar.selectbox("Family history with overweight", ["no", "yes"], index=0, key="family_history_with_overweight")
    favc = st.sidebar.selectbox("Frequent high-calorie food (FAVC)", ["no", "yes"], index=0, key="FAVC")
    fcvc = st.sidebar.selectbox("Vegetable consumption (FCVC) [1-3]", [1,2,3], index=1, key="FCVC")
    ncp = st.sidebar.selectbox("Number of main meals (NCP)", [1,2,3], index=2, key="NCP")
    caec = st.sidebar.selectbox("Snacking between meals (CAEC)", ["no","Sometimes","Frequently","Always"],
                                index=0, key="CAEC")
    calc = st.sidebar.selectbox("Alcohol consumption (CALC)", ["no","Sometimes","Frequently"], index=0, key="CALC")
    scc = st.sidebar.selectbox("Calorie consumption monitoring (SCC)", ["no","yes"], index=0, key="SCC")
    mtrans = st.sidebar.selectbox("Transportation (MTRANS)",
                                  ["Automobile","Walking","Public_Transportation","Bike","Motorbike","No_transport"],
                                  index=1, key="MTRANS")

    user_input = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "Gender": gender,
        "SMOKE": smoke,
        "family_history_with_overweight": family_over,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "CALC": calc,
        "SCC": scc,
        "MTRANS": mtrans
    }

    # show BMI
    bmi_val = bmi_from_row({"Height": height, "Weight": weight})
    st.subheader("Calculated BMI")
    if not np.isnan(bmi_val):
        st.metric("BMI", f"{bmi_val}")
    else:
        st.write("BMI could not be calculated (missing or invalid height/weight).")

    # Predict single
    if st.button("Predict (single)"):
        # normalize categories to match training
        user_input = normalize_categories(user_input)

        X = prepare_input_df(user_input, training_features)

        # If model not loaded: fallback rule based on BMI
        if model is None:
            prob = 1.0 if (not np.isnan(bmi_val) and bmi_val >= 30) else 0.0
            pred = 1 if prob >= 0.5 else 0
            st.warning("No saved model loaded; using BMI threshold fallback (BMI >= 30 -> obese).")
            st.write({"obesity_prob": float(prob), "obesity_pred": int(pred)})
        else:
            try:
                proba, preds = predict_with_model(model, X)
                st.success(f"Predicted obesity probability: {proba[0]:.6f}")
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

        # compute BMI if possible
        if "Height" in df_upload.columns and "Weight" in df_upload.columns:
            df_upload["BMI"] = df_upload.apply(lambda r: bmi_from_row({"Height": r["Height"], "Weight": r["Weight"]}), axis=1)

        if st.button("Predict (batch)"):
            # ensure required cols exist in correct order
            X_batch = df_upload.copy()
            for c in training_features:
                if c not in X_batch.columns:
                    X_batch[c] = np.nan

            if model is None:
                df_upload["obesity_prob"] = df_upload["BMI"].apply(lambda x: 1.0 if x >= 30 else 0.0)
                df_upload["obesity_pred"] = df_upload["obesity_prob"].apply(lambda x: int(x >= 0.5))
                st.warning("No saved model loaded; used BMI-threshold fallback.")
                st.dataframe(df_upload.head())
            else:
                try:
                    # Normalize categories in dataframe (best-effort)
                    # lowercase -> map to training categories for CALC and CAEC
                    def normalize_df_categories(df):
                        df = df.copy()
                        if "CALC" in df.columns:
                            df["CALC"] = df["CALC"].astype(str).str.strip().str.lower().map({
                                "no":"no","sometimes":"Sometimes","frequently":"Frequently"
                            }).fillna(df["CALC"])
                        if "CAEC" in df.columns:
                            df["CAEC"] = df["CAEC"].astype(str).str.strip().str.lower().map({
                                "no":"no","sometimes":"Sometimes","frequently":"Frequently","always":"Always"
                            }).fillna(df["CAEC"])
                        return df

                    X_norm = normalize_df_categories(X_batch)
                    proba, preds = predict_with_model(model, X_norm[training_features])
                    df_upload["obesity_prob"] = proba
                    df_upload["obesity_pred"] = preds
                    st.success("Predictions added to the uploaded dataframe.")
                    st.dataframe(df_upload.head())
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

            csv = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# ---------------------------
# Model explanation / Feature importance
# ---------------------------
st.markdown("---")
st.subheader("Model explanation / Feature importance")

if model is None:
    st.info("No model loaded: feature importance not available.")
else:
    try:
        # If logistic regression pipeline
        if hasattr(model.named_steps['clf'], "coef_"):
            coef = model.named_steps['clf'].coef_[0]
            # try to get feature names (best-effort)
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
            feat_table = feat_table.sort_values(by="coef", key=abs, ascending=False).head(10)
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
