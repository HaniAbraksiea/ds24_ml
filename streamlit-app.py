
import streamlit as st
import numpy as np
import joblib
import cv2
from streamlit_drawable_canvas import st_canvas

# Ladda den sparade rf modellen 
model = joblib.load("best_rf_model.pkl")


st.title("Rita en siffra och låt modellen gissa!")

# Skapa en canvas för att rita siffror
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Prediktera") and model is not None:
    if canvas_result.image_data is not None:
        # Konvertera bilddata
        img_array = np.array(canvas_result.image_data)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)  # Gråskala
        img_array = cv2.resize(img_array, (28, 28))  # Resiza
        img_array = img_array / 255.0  # Normalisera
        img_array = img_array.flatten().reshape(1, -1)  # Platta ut

        # Skala data
        # img_array_scaled = scaler.transform(img_array)

        # Gör prediktion
        prediction = model.predict(img_array)
        predicted_label = prediction[0]

        # Visa resultat
        st.subheader("Prediktion")
        st.write(f"**Modellen predikterar:** {predicted_label}")

#cd OneDrive\Dokument\Data_scientist_EC_utbildning\05_ml\ds24_ml-main\kunskapskontroll_2\Kunskapskontroll_2_Hani_Abraksiea
#streamlit run streamlit-app.py


