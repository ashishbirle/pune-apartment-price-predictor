import streamlit as st
import pandas as pd
import pickle


model_lr = pickle.load(open("models/model_lr.pkl", "rb"))
model_rf = pickle.load(open("models/model_rf.pkl", "rb"))
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

model_choice = st.selectbox(
    "Choose Model",
    [
        "Linear Regression",
        "Random Forest"
    ]
)



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

    if model_choice == "Linear Regression":
        prediction = model_lr.predict(input_df)[0]
    
    else:
        prediction = model_rf.predict(input_df)[0]

    st.success(
        f"Estimated Price: Rs {prediction:,.0f}"
    )