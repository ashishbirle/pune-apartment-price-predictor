import streamlit as st
import pandas as pd
import pickle


model = pickle.load(open("models/house_price_model.pkl", "rb"))
area_encoder = pickle.load(open("models/area_encoder.pkl", "rb"))
yesno_encoder = pickle.load(open("models/yesno_encoder.pkl", "rb"))

st.title("Pune Apartment Price Predictor")

area = st.selectbox(
    "Area",
    ["Baner", "Nigdi", "Punawale", "Wakad"]
)

bhk = st.selectbox("BHK", [1,2,3])

parking = st.checkbox("Parking")
balcony = st.checkbox("Balcony")
gym = st.checkbox("Gym")
security = st.checkbox("Security")
swimming_pool = st.checkbox("Swimming Pool")

if st.button("Predict Price"):
    area_encoded = area_encoder.transform([area])[0]

    input_df = pd.DataFrame([{
        "area": area_encoded,
        "bhk": bhk,
        "parking": int(parking),
        "balcony": int(balcony),
        "gym": int(gym),
        "security": int(security),
        "swimming_pool": int(swimming_pool)
    }])

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Price: Rs {prediction:,.0f}"
    )