import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Pricing Intelligence System")

st.write("Upload your dataset to analyze pricing")

# File upload
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Data Preview")
    st.write(df.head())

    # -------------------------
    # ELASTICITY CALCULATION
    # -------------------------
    st.subheader("Price Elasticity")

    df['demand'] = 1

    grouped = df.groupby('price').agg({'demand':'sum'}).reset_index()

    grouped = grouped[(grouped['price'] > 0) & (grouped['demand'] > 0)]
    grouped = grouped[grouped['price'] < 1000]

    grouped['log_price'] = np.log(grouped['price'])
    grouped['log_demand'] = np.log(grouped['demand'])

    model = LinearRegression().fit(grouped[['log_price']], grouped['log_demand'])
    elasticity = model.coef_[0]

    st.write(f"Elasticity: {elasticity:.2f}")

    # -------------------------
    # GRAPH
    # -------------------------
    st.subheader("Price vs Demand")

    fig, ax = plt.subplots()
    ax.scatter(grouped['price'], grouped['demand'])
    ax.set_xlabel("Price")
    ax.set_ylabel("Demand")
    ax.set_title("Price vs Demand")

    st.pyplot(fig)

    # -------------------------
    # DISCOUNT OPTIMIZATION
    # -------------------------
    st.subheader("Best Discount Strategy")

    discounts = [0, 0.1, 0.2, 0.3]
    results = []

    for d in discounts:
        grouped['new_price'] = grouped['price'] * (1 - d)
        grouped['predicted_demand'] = grouped['demand'] * (grouped['new_price'] / grouped['price']) ** elasticity
        grouped['revenue'] = grouped['new_price'] * grouped['predicted_demand']

        total_revenue = grouped['revenue'].sum()
        results.append((d, total_revenue))

    best = max(results, key=lambda x: x[1])

    st.write(f"Best Discount: {best[0]*100:.0f}%")
    st.write(f"Max Revenue: {best[1]:,.2f}")