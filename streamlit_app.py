import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# ------------- Model Training -------------
def train_models(df):
    st.subheader("🔧 Training Models...")

    df_encoded = pd.get_dummies(df, columns=["zipcode", "state"])
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    # Scale inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale target
    y_max = y.max()
    y_scaled = y / y_max
    joblib.dump(y_max, "y_max.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "linear_model.pkl")

    # ANN Model
    ann = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    ann.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    ann.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stop], verbose=1)

    ann.save("ann_model.keras")
    joblib.dump(scaler, "scaler.pkl")

    st.success("✅ Models trained and saved!")


# ------------- User Prediction -------------
def user_prediction(df, state):
    st.subheader("🤖 Predict House Price (Custom Input)")

    user_data = {
        "bedrooms": st.number_input("Bedrooms", 1, 10, 3),
        "bathrooms": st.number_input("Bathrooms", 1, 5, 2, step=1, format="%d"),
        "sqft_living": st.number_input("Living Area (sqft)", 300, 10000, 1500),
        "sqft_lot": st.number_input("Lot Size (sqft)", 500, 20000, 3000),
        "yr_built": st.number_input("Year Built", 1900, 2025, 2000),
        "zipcode": st.selectbox("Zipcode", sorted(df[df["state"] == state]["zipcode"].unique())),
        "state": state
    }

    df_input = pd.DataFrame([user_data])
    df_encoded = pd.get_dummies(pd.concat([df, df_input], ignore_index=True), columns=["zipcode", "state"])
    df_input_encoded = df_encoded.iloc[[-1]]
    df_encoded_full = df_encoded.drop("price", axis=1, errors='ignore')

    # Align columns
    X = df_encoded_full
    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X)

    # Load models
    lr_model = joblib.load("linear_model.pkl")
    ann_model = load_model("ann_model.keras")
    y_max = joblib.load("y_max.pkl")

    lr_pred_scaled = lr_model.predict(X_scaled[-1].reshape(1, -1))[0]
    lr_pred = lr_pred_scaled * y_max
    ann_pred_scaled = ann_model.predict(X_scaled[-1].reshape(1, -1))[0][0]
    ann_pred = ann_pred_scaled * y_max

    st.success(f"💡 Linear Regression Prediction: ${lr_pred:,.2f}")
    st.success(f"⚙️ Neural Network Prediction: ${ann_pred:,.2f}")


# ------------- Error Analysis -------------
def error_analysis(df):
    st.subheader("🧪 Model Error Comparison")

    df_encoded = pd.get_dummies(df, columns=["zipcode", "state"])
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    scaler = joblib.load("scaler.pkl")
    y_max = joblib.load("y_max.pkl")
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    lr_model = joblib.load("linear_model.pkl")
    ann_model = load_model("ann_model.keras")

    lr_preds_scaled = lr_model.predict(X_test)
    ann_preds_scaled = ann_model.predict(X_test).flatten()

    lr_preds = lr_preds_scaled * y_max
    ann_preds = ann_preds_scaled * y_max

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    ann_rmse = np.sqrt(mean_squared_error(y_test, ann_preds))

    st.write(f"📉 Linear Regression RMSE: {lr_rmse:,.2f}")
    st.write(f"📈 ANN RMSE: {ann_rmse:,.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, lr_preds, label='Linear Regression', alpha=0.6)
    ax.scatter(y_test, ann_preds, label='ANN', alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Prices")
    ax.legend()
    st.pyplot(fig)


# ------------- Model Recommendation -------------
def model_recommendation(df, state):
    st.subheader("📌 Model Recommendation")

    df_encoded = pd.get_dummies(df, columns=["zipcode", "state"])
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    scaler = joblib.load("scaler.pkl")
    y_max = joblib.load("y_max.pkl")
    X_scaled = scaler.transform(X)

    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2)

    lr_model = joblib.load("linear_model.pkl")
    ann_model = load_model("ann_model.keras")

    lr_preds_scaled = lr_model.predict(X_test)
    ann_preds_scaled = ann_model.predict(X_test).flatten()

    lr_preds = lr_preds_scaled * y_max
    ann_preds = ann_preds_scaled * y_max

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    ann_rmse = np.sqrt(mean_squared_error(y_test, ann_preds))

    better_model = "Linear Regression" if lr_rmse < ann_rmse else "Artificial Neural Network"
    st.success(f"✅ Recommendation: **{better_model}** is more accurate for predicting prices in {state}.")
    st.write(f"Linear RMSE: {lr_rmse:,.2f} | ANN RMSE: {ann_rmse:,.2f}")


# ------------- State and Zip Filtering -------------
def filter_by_state_zip(df):
    st.subheader("📍 Filter by State and Zipcode")

    state = st.selectbox("Choose a State to filter", sorted(df["state"].unique()))
    zipcodes = sorted(df[df["state"] == state]["zipcode"].unique())
    zipcode = st.selectbox("Choose a Zipcode", zipcodes)

    filtered = df[(df["state"] == state) & (df["zipcode"] == zipcode)]
    st.write(f"🏘️ Showing {len(filtered)} listings for {state} - {zipcode}")
    st.dataframe(filtered)


# ------------- Data Visualization -------------
def data_visualization(df):
    st.subheader("📊 Data Visualization Dashboard")

    st.write("### Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], bins=50, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.write("### Correlation Heatmap")
    df_encoded = pd.get_dummies(df.drop(["zipcode", "state"], axis=1), drop_first=True)
    corr = df_encoded.corr()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.write("### Price vs Sqft Living")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="sqft_living", y="price", data=df, hue="state", ax=ax3)
    st.pyplot(fig3)


# ---------------- Main app ----------------
def main():
    st.title("🏠 House Price Prediction AI")

    df = pd.read_csv("realistic_dummy_real_estate_data.csv")

    # Limit to only 4 states
    selected_states = sorted(df["state"].unique())[:4]
    state = st.selectbox("Select State", selected_states)

    option = st.sidebar.radio("Choose Option", (
        "Train Models",
        "State and Zip Code Based Filtering",
        "User Input for Custom Predictions",
        "Data Visualization Dashboard",
        "Error Analysis",
        "Model Selection Recommendation"
    ))

    if option == "Train Models":
        train_models(df)
    elif option == "State and Zip Code Based Filtering":
        filter_by_state_zip(df)
    elif option == "User Input for Custom Predictions":
        user_prediction(df, state)
    elif option == "Data Visualization Dashboard":
        data_visualization(df)
    elif option == "Error Analysis":
        error_analysis(df)
    elif option == "Model Selection Recommendation":
        model_recommendation(df, state)


if __name__ == "__main__":
    main()
