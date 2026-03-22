import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load trained model
model = load_model("mnist_cnn_model.h5")

st.title("🧠 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (28x28 or any size)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to array
    image_array = np.array(image)
    
    # Normalize
    image_array = image_array / 255.0
    
    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)
    
    # Prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    
    # Display image and result
    st.image(image, caption="Uploaded Image", width=150)
    st.success(f"Predicted Digit: {predicted_digit}")
