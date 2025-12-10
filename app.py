import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Load Model
try:
    with open('asl_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'asl_model.pkl' not found. Please upload it to the repo.")

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 3. Define the Processing Class
class HandGestureProcessor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process Hand
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        prediction_text = "Waiting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract Features (Same logic as training)
                landmarks = []
                for i in range(len(hand_landmarks.landmark)):
                    landmarks.append(hand_landmarks.landmark[i].x)
                    landmarks.append(hand_landmarks.landmark[i].y)
                
                wrist_x = landmarks[0]
                wrist_y = landmarks[1]
                
                normalized_landmarks = []
                for i in range(0, len(landmarks), 2):
                    normalized_landmarks.append(landmarks[i] - wrist_x)
                    normalized_landmarks.append(landmarks[i+1] - wrist_y)

                # Predict
                try:
                    prediction = model.predict([normalized_landmarks])[0]
                    prediction_text = f"Prediction: {prediction}"
                except Exception as e:
                    prediction_text = "Error"

        # Draw Prediction on Frame
        cv2.rectangle(img, (0,0), (640, 60), (0,0,0), -1)
        cv2.putText(img, prediction_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

# 4. Streamlit Layout
st.title("🤟 Real-Time ASL Detector")
st.write("Turn on your webcam and show a hand gesture!")

# Start WebRTC Stream
webrtc_streamer(
    key="example", 
    video_transformer_factory=HandGestureProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
