import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("mnist_cnn_model.h5")

def predict_digit(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    return int(np.argmax(prediction))

# Gradio UI
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="🧠 MNIST Digit Recognition"
)

interface.launch()