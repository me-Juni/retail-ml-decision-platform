import streamlit as st
import requests


st.title("☕ Retail Demand Decision Dashboard")

st.write("Predict tomorrow demand and staffing needs")

store_id = st.number_input("Store ID", min_value=1, max_value=10, value=1)
product_id = st.number_input("Product ID", min_value=1, max_value=50, value=1)

if st.button("Get Prediction"):

    url = f"http://127.0.0.1:8000/predict/{store_id}/{product_id}"
    response = requests.get(url).json()

    if "error"  in response:
        st.error(response["error"])
    else:
        predicted = response["predicted_units_sold"]

        # simple decision logic
        if predicted < 20:
            staff = 2
        elif predicted < 40:
            staff = 3
        elif predicted < 60:
            staff = 4
        else: 
            staff = 5
        
        inventory = int(predicted * 1.2)

        st.subheader("📈 Forecast")
        st.write(f"Expected Demand: **{predicted:.2f} units**")

        st.subheader("🧑‍🍳 Staffing Recommendation")
        st.write(f"Recommended staff: **{staff} employees**")

        st.subheader("📦 Inventory Recommendation")
        st.write(f"Prepare at least **{inventory} units**")