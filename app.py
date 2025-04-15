import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model(r"C:\Users\rpriy\Desktop\WORKING ON THIS SHIT\8th sem shit\FINAL PROJECT 4 with streamlit\facial_expression_model.h5")

# Define emotions list (update with your actual emotion classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to make predictions
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_resized = np.reshape(roi_resized, (1, 48, 48, 1))
        roi_resized = roi_resized / 255.0
        prediction = model.predict(roi_resized)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion, roi_color
    return None, image

# Set up the Streamlit UI
st.title("Facial Expression Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded image to OpenCV format
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Predict emotion
    emotion, result_image = predict_emotion(image)
    
    if emotion:
        st.image(result_image, channels="BGR", caption=f"Emotion: {emotion}", use_column_width=True)
    else:
        st.image(result_image, caption="No face detected", use_column_width=True)
