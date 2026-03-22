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

    prediction = model.predict(image)
    probs = prediction[0]

    return {str(i): float(probs[i]) for i in range(10)}

# Custom CSS for better UI
custom_css = """
body {background-color: #0f172a;}
h1 {text-align: center; color: #38bdf8;}
.gr-button {background: #2563eb; color: white; border-radius: 8px;}
"""

# UI Layout
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 Handwritten Digit Recognition  
    ### Upload an image and let the AI predict the digit (0–9)
    """)

    with gr.Row():
        image_input = gr.Image(type="pil", label="📤 Upload Digit Image")
        output = gr.Label(label="🔮 Prediction")

    submit_btn = gr.Button("🚀 Predict")

    submit_btn.click(fn=predict_digit, inputs=image_input, outputs=output)

    gr.Markdown("💡 Tip: Use white digit on black background for best results")

demo.launch()