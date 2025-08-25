
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Exam Score Prediction", page_icon="🎓", layout="wide")

st.title("🎓 Student Exam Score Prediction")
st.write("Predict exam scores with Linear Regression. Upload your CSV or use a local file named **StudentPerformanceFactors.csv** in the same folder as this app.")

# Sidebar controls
st.sidebar.header("Settings")
data_source = st.sidebar.radio("Choose data source", ["Use local file", "Upload CSV"])
local_path = st.sidebar.text_input("Local CSV filename", value="StudentPerformanceFactors.csv")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
n_splits = st.sidebar.slider("k-Fold splits (k)", 3, 10, 5)
top_n = st.sidebar.slider("Top features to show", 5, 30, 20)

# Load data
df = None
if data_source == "Use local file":
    try:
        df = pd.read_csv(local_path)
    except Exception as e:
        st.error(f"Couldn't read `{local_path}`: {e}")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.info("👉 Provide a dataset to begin. It must contain a numeric column named `Exam_Score`.")
    st.stop()

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())
st.write(f"Rows: **{len(df)}** | Columns: **{df.shape[1]}**")

# Basic cleaning
df = df.drop_duplicates()

# Handle missing values: fill categorical with mode, numeric with median
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == "O":
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    else:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

if "Exam_Score" not in df_clean.columns:
    st.error("The dataset must include an `Exam_Score` column to be the target variable.")
    st.stop()

# Encode categoricals
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# Split
X = df_encoded.drop("Exam_Score", axis=1)
y = df_encoded["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
mse_tt = mean_squared_error(y_test, y_pred)
rmse_tt = np.sqrt(mse_tt)
r2_tt = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MSE (test)", f"{mse_tt:.3f}")
col2.metric("RMSE (test)", f"{rmse_tt:.3f}")
col3.metric("R² (test)", f"{r2_tt:.3f}")

# k-Fold CV
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
cv_mse = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")

st.subheader("🔁 k-Fold Cross Validation")
st.write("R² scores per fold:", np.round(cv_r2, 3))
st.write("Average R²:", f"{np.mean(cv_r2):.3f}")
st.write("Average MSE:", f"{np.mean(cv_mse):.3f}")

# Visualizations
st.subheader("📈 Visualizations")

# Predicted vs Actual
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.7)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
ax1.set_xlabel("Actual Exam_Score")
ax1.set_ylabel("Predicted Exam_Score")
ax1.set_title("Actual vs Predicted (Test Set)")
st.pyplot(fig1)

# Residuals histogram
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
ax2.hist(residuals, bins=20)
ax2.set_xlabel("Residual")
ax2.set_title("Residuals Distribution")
st.pyplot(fig2)

# Feature coefficients (top N by absolute value)
coef = pd.Series(model.coef_, index=X.columns)
coef_abs_sorted = coef.abs().sort_values(ascending=False).head(top_n)
coef_plot = coef[coef_abs_sorted.index]  # keep signs

fig3, ax3 = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
ax3.barh(coef_plot.index, coef_plot.values)
ax3.axvline(0, linestyle="--")
ax3.set_xlabel("Coefficient")
ax3.set_title(f"Top {top_n} Features by |Coefficient|")
ax3.invert_yaxis()
st.pyplot(fig3)

st.caption("Tip: Use the sidebar to adjust test size, number of CV folds, and how many top features to display.")
