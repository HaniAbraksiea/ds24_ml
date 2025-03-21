import streamlit as st
import numpy as np
import joblib
import cv2
import os
import gdown
from streamlit_drawable_canvas import st_canvas

# Cachea modellen så att den inte laddas om varje gång
@st.cache_resource
def load_cached_model():
    # Google Drive fil-ID för din modell
    file_id = '1DftgO3YjlxBxLOP4Nsy6AsNH7_1W1AAz'  # Nytt Fil-ID från din Google Drive-länk
    url = f'https://drive.google.com/uc?export=download&id={file_id}'  # Omvandla till rätt URL
    output = 'best_rf_model.pkl'

    # Ladda ner modellen om den inte finns lokalt
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    # Ladda modellen från filen
    model = joblib.load(output)
    return model

model = load_cached_model()

st.title("Rita en siffra mellan 0-9 och låt modellen gissa! (Draw a digit between 0-9 and let the model guess!)")

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
        try:
            # Konvertera bilddata
            img_array = np.array(canvas_result.image_data)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)  # Gråskala
            img_array = cv2.resize(img_array, (28, 28))  # Resiza till 28x28
            img_array = img_array / 255.0  # Normalisera
            img_array = img_array.flatten().reshape(1, -1)  # Platta ut

            # Gör prediktion
            prediction = model.predict(img_array)
            predicted_label = prediction[0]

            # Visa resultat
            st.subheader("Prediktion")
            st.write(f"**Modellen predikterar:** {predicted_label}")
        except Exception as e:
            st.error(f"Fel vid bildbehandling: {e}")
    else:
        st.error("Ingen bild ritad på duken.")
