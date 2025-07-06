# House Price Prediction AI

This is a web application built with Streamlit that predicts house prices using two models: Linear Regression and Artificial Neural Networks (ANN).

Users can explore housing data, train models, visualize trends, and make predictions based on custom inputs. The app is designed to be interactive and educational, making it easier to understand how different features affect property prices.

---

## Features

- Train and compare two machine learning models (Linear Regression and ANN)
- Filter listings by state and ZIP code
- Make custom price predictions based on user input (bedrooms, size, year, etc.)
- Visualize price distributions, correlations, and feature relationships
- Compare model performance using error metrics (RMSE)
- Get a recommendation on which model works better for a given dataset

---

## Files Included

- `app.py` – Main application script
- `realistic_dummy_real_estate_data.csv` – Sample dataset
- `linear_model.pkl`, `ann_model.keras` – Trained model files
- `scaler.pkl`, `y_max.pkl` – Preprocessing tools for scaling inputs and outputs
- `README.txt` – Project documentation

---

## Getting Started

To run the project locally:

1. Install the required Python libraries:

```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn tensorflow joblib
```

2. Launch the app using Streamlit:

```
streamlit run app.py
```

This will open the web app in your browser, where you can interact with the various features.

---

## About the Project

This project was developed as a part of a learning journey in AI and machine learning. It combines both classical and deep learning models in a simple, visual interface to make predictions on housing data.

It’s a great starting point for anyone interested in applying machine learning to real-world problems like real estate pricing.