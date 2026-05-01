# 🔢 Handwritten Digit Recognition (Deployed on Hugging Face)

## 📌 Overview

This project focuses on building a machine learning/deep learning model to recognize **handwritten digits (0–9)**. The model is trained on image data and deployed as an interactive web application using **Hugging Face Spaces**, allowing users to draw digits and get real-time predictions.

---

## 🎯 Objectives

* Build a robust model for handwritten digit classification
* Perform data preprocessing and normalization
* Train and evaluate deep learning models
* Deploy the model as an interactive web app
* Enable real-time predictions via user input

---

## 📊 Dataset

* **Dataset Used**: MNIST (Modified National Institute of Standards and Technology)
* **Total Images**: 70,000

  * Training: 60,000
  * Testing: 10,000
* **Image Size**: 28 × 28 grayscale

---

## 🔍 Data Preprocessing

* Normalized pixel values (0–255 → 0–1)
* Reshaped images for model input
* One-hot encoded target labels (for deep learning models)

---

## 🤖 Model Architecture

### Deep Learning Model (CNN)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
```

### 💡 Why CNN?

* Captures spatial patterns in images
* Highly effective for image classification tasks
* Automatically extracts features

---

## ⚙️ Training Details

* Loss Function: `categorical_crossentropy`
* Optimizer: `Adam`
* Epochs: 5–10
* Batch Size: 32

---

## 📈 Model Performance

* **Test Accuracy**: ~98%–99%
* High precision and recall across all digit classes

---

## 🧪 Evaluation

* Confusion matrix shows minimal misclassification
* Most errors occur between visually similar digits (e.g., 5 vs 3, 9 vs 4)

---

## 🚀 Deployment (Hugging Face Spaces)

The model is deployed using **Gradio** on Hugging Face Spaces.

👉 **Live Demo**:
🔗 *[Add your Hugging Face link here]*

---

## 🖥️ App Features

* Draw digits on canvas
* Real-time prediction
* User-friendly interface
* Fast inference

---

## 🧠 How It Works

1. User draws a digit on canvas
2. Image is preprocessed (resized to 28×28, normalized)
3. Model predicts the digit
4. Output displayed instantly

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Matplotlib
* Gradio
* Hugging Face Spaces

---

## 📂 Project Structure

```
├── app.py              # Gradio app
├── model.h5            # Trained model
├── requirements.txt    # Dependencies
├── README.md
```

---

## 🔮 Future Improvements

* Improve accuracy using deeper CNN architectures
* Add probability visualization for predictions
* Support multi-digit recognition
* Deploy mobile-friendly version

---

## 📌 Key Learnings

* CNNs are highly effective for image classification
* Proper preprocessing significantly impacts performance
* Deployment makes ML models usable in real-world scenarios

---

## 👤 Author

**Vrund Patel**
Aspiring AI Engineer | Machine Learning Enthusiast
