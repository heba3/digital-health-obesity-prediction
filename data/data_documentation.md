# Data Documentation

This document provides a unified overview of the datasets used in the **Obesity Risk Prediction (Hybrid Model)** project. It includes documentation for both the **Obesity dataset** and the **Fitbit dataset**.

---

# 1. Obesity Dataset Documentation

## 1.1 Dataset Overview
The obesity dataset contains demographic, behavioral, and lifestyle information used to classify individuals into obesity categories.

### Dataset Statistics
- Total rows: 2111  
- Total columns: 18  
  - 17 input features  
  - 1 target variable (`Obesity_Level`)

### Target Variable  
`Obesity_Level` is a multi-class categorical variable containing the following labels:
- Insufficient_Weight  
- Normal_Weight  
- Overweight_Level_I  
- Overweight_Level_II  
- Obesity_Type_I  
- Obesity_Type_II  
- Obesity_Type_III  

---

## 1.2 Variable Dictionary

| Feature | Type | Description | Notes |
|--------|------|-------------|-------|
| Age | Numeric | Age of the individual | Range: 14–61 |
| Gender | Categorical | Male / Female | One-hot encoded |
| Height | Numeric | Height in meters (m) | Converted from cm when needed |
| Weight | Numeric | Weight in kilograms | Used to compute BMI |
| BMI | Numeric | Body Mass Index | Engineered feature |
| family_history_with_overweight | Categorical | Family obesity history | Yes / No |
| FAVC | Categorical | Frequent high-calorie food | Yes / No |
| FCVC | Numeric | Vegetable consumption | Scale: 1–3 |
| NCP | Numeric | Main meals per day | Scale: 1–4 |
| SMOKE | Categorical | Smoking habit | Yes / No |
| CH2O | Numeric | Daily water intake | Typically 1–3 |
| SCC | Categorical | Calories monitored | Yes / No |
| FAF | Numeric | Physical activity | Scale: 0–3 |
| TUE | Numeric | Tech usage time | Scale: 0–2 |
| CALC | Categorical | Alcohol consumption | No / Sometimes / Frequently |
| CAEC | Categorical | Eating between meals | No / Sometimes / Frequently / Always |
| MTRANS | Categorical | Transportation mode | Walking, Automobile, Bike, etc. |
| Obesity_Level | Target | Multi-class label | Used for training |

---

## 1.3 Preprocessing Notes

### Categorical Encoding
One-Hot Encoding applied to:
- Gender  
- CALC  
- FAVC  
- SCC  
- SMOKE  
- family_history_with_overweight  
- CAEC  
- MTRANS  

### Numerical Scaling
`StandardScaler` applied on:
- Age  
- Height  
- Weight  
- FCVC  
- NCP  
- CH2O  
- FAF  
- TUE  

### Engineered Features
BMI is computed as:
```
BMI = Weight / (Height^2)
```

### Missing Values
- The dataset originally contains no missing values.  
- Verified during EDA.

### Outlier Analysis
Outliers checked using:
- Z-score  
- Boxplots  
- Distribution plots  
- Logical constraints (e.g., realistic BMI range)

---

## 1.4 Assumptions
- Height values are correctly convertible into meters.  
- All categorical answers follow defined formats.  
- Self-reported lifestyle values are assumed reasonably accurate.  
- No missing values were present originally.

---

# 2. Fitbit Dataset Documentation

## 2.1 Dataset Overview
The Fitbit dataset contains daily physiological and behavioral measurements collected from wearable devices.

Since this dataset does **not** share a user identifier with the obesity dataset, integration is performed using **population-level augmentation**.

---

## 2.2 Dataset Statistics

| Item | Value |
|------|-------|
| Total rows | Depends on collection period |
| Total columns | Around 4–6 |
| Granularity | Daily records |
| Link to Obesity dataset | No shared ID available |

---

## 2.3 Variable Dictionary (Fitbit)

| Feature | Type | Description | Notes |
|--------|------|-------------|-------|
| date | Date | Measurement date | Converted to datetime |
| steps | Numeric | Daily step count | Typical range: 0–20,000+ |
| sleep_points_percentage | Numeric | Sleep quality score | Normalized 0–1 |
| stress_score | Numeric | Daily stress level | Scale: 0–100 |
| rmssd | Numeric | Heart Rate Variability (HRV) | Milliseconds |
| user_id | Optional | Fitbit user identifier | Often missing |

---

## 2.4 Preprocessing Notes

### Numeric Conversion
All numerical fields are converted with:
```
pd.to_numeric(..., errors="coerce")
```

### Normalization
If sleep score is 0–100:
```
sleep_points_percentage /= 100
```

### Outlier Handling
- Steps > 100,000 are set to NaN  
- RMSSD < 5 or > 200 flagged  
- Stress > 100 considered invalid  

### Missing Values
- HRV and sleep quality may contain many missing values  
- Imputed or averaged during aggregation

---

## 2.5 Aggregation Strategy

### A. Per-user Aggregation (if `user_id` exists)
Aggregations include:
- Mean steps  
- Mean sleep score  
- Mean stress score  
- Mean RMSSD  

Saved as:  
`data/processed/per_user_fitbit_aggregates.csv`

### B. Population-Level Aggregation (used in this project)
Since no user mapping exists, global averages are added to every row in the obesity dataset:

| New Feature | Description |
|-------------|-------------|
| fit_steps_mean | Mean steps across Fitbit dataset |
| fit_sleep_points_percentage_mean | Mean sleep score |
| fit_stress_score_mean | Mean stress |
| fit_rmssd_mean | Mean HRV |

Saved as:  
`data/processed/obesity_fitbit_augmented.csv`

---

# 3. Versioning
Changes to the following should be documented in future versions:
- Dataset sources  
- Preprocessing logic  
- Feature engineering  
- Aggregation strategy  
- Augmented datasets  
