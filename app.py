import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("final_mobilenetv3.keras")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Recommendations
disease_info = {
    "Tomato___Early_blight": "Remove infected leaves. Apply approved fungicide. Avoid overhead watering.",
    "Tomato___Late_blight": "Use resistant varieties and fungicide. Remove infected plants.",
    "Tomato___healthy": "Plant appears healthy. Continue normal care."
}

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to identify the disease.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    disease = class_names[class_index]

    st.success(
        f"Disease: {disease.replace('___', ' ').replace('_', ' ')}"
    )

    st.info(f"Confidence: {confidence:.2f}%")

    if disease in disease_info:
        st.subheader("Recommendation")
        st.write(disease_info[disease])
