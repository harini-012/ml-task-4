import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

st.title("Ridge Regression for Housing Price Prediction")

df = pd.read_csv("housing.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Data Exploration")
st.write(df.isnull().sum())
st.write(df.describe())

fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Data Preparation")

df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

X = df[['longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms',
        'population', 'households', 'median_income']]

y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.subheader("Ridge Regression Model")

alpha = st.slider("Select alpha value", 0.1, 10.0, 1.0)

model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)

st.subheader("Model Evaluation")

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("MSE:", mse)
st.write("R2 Score:", r2)

st.subheader("Feature Coefficients")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.write(coef_df)

st.subheader("Actual vs Predicted")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs Predicted")

ax2.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])

st.pyplot(fig2)