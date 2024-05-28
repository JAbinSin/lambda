import cv2
import time
import logging
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
import threading
import xgboost as xgb
import pandas as pd
import numpy as np
from flask import session

# Disable logging for YOLOv8
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Get current date
current_date = datetime.now().strftime('%d-%m-%Y')

# Setup logging
detection_logger = logging.getLogger('response')
detection_logger.setLevel(logging.INFO)

# Create file handler and set level to INFO
file_handler = logging.FileHandler(f'logs/{current_date}.log')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger
detection_logger.addHandler(file_handler)

# Load the pre-trained YOLOv8 pose estimation model
pose_model = YOLO("yolov8n-pose.pt")

# Load the XGBoost model
model_xgb = xgb.Booster()
model_xgb.load_model('training/video/level-3-output/model_weights.xgb')

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to reload LBPH trained model
def reload_lbph_trained_model():
    global lbph_recognizer
    try:
        lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
        lbph_recognizer.read("trained_models/lbph_trained_model.yml")
    except cv2.error as e:
        print("Warning: LBPH recognizer could not be loaded. Error:", e)

# Initial load of the LBPH recognizer
reload_lbph_trained_model()

# Database connection
def get_db_connection():
    return sqlite3.connect('database/database.db')

# Email notification function
def send_email_notification(name, subject, message, email):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = 'lambda54312@gmail.com'
    msg['To'] = email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('lambda54312@gmail.com', 'yowc qcuc dgxg ngol')
            server.send_message(msg)
        print(f"Email sent to {email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Face detection and pose estimation function
def detect_faces_and_poses(frame, detection_times, last_detection_time, face_last_seen, email):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    results = pose_model(frame, verbose=False)
    frame_with_results = results[0].plot(boxes=False)

    for r in results:
        bound_box = r.boxes.xyxy
        conf = r.boxes.conf.tolist()
        keypoints = r.keypoints.xyn.tolist()

        for index, box in enumerate(bound_box):
            if conf[index] > 0.75:
                x1, y1, x2, y2 = box.tolist()
                data = {}

                # Initialize the x and y lists for each possible key
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]

                df = pd.DataFrame(data, index=[0])
                dmatrix = xgb.DMatrix(df)
                behavior = model_xgb.predict(dmatrix)
                binary_predictions = np.argmax(behavior)  # Get the class with the highest probability

                # Determine action label
                action_labels = ['drunk', 'kicking', 'punching', 'running', 'seating', 'smashing', 'squating', 'standing', 'walking']
                action_label = action_labels[binary_predictions]

                # Draw action label in upper left corner
                cv2.putText(frame_with_results, action_label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False, detector_backend="opencv")
            dominant_emotion = result[0].get("dominant_emotion", "Unknown")
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print("Error analyzing emotions:", e)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if lbph_recognizer:
            label, confidence = lbph_recognizer.predict(roi_gray)
            confidence = 100 - confidence

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM faces WHERE id = ?", (label,))
                face_name = cursor.fetchone()

            if face_name and confidence >= 50:
                name = face_name[0]
                cv2.putText(frame_with_results, f"{name}: {confidence:.2f}%", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                current_time = time.time()
                if name not in detection_times:
                    detection_times[name] = current_time
                else:
                    if current_time - detection_times[name] >= 5 and (current_time - last_detection_time.get(name, 0)) >= 5:
                        subject = "Face Detected"
                        message = "Face Detected from " + name
                        send_email_notification(name, subject, message, email)
                        last_detection_time[name] = current_time
                face_last_seen[name] = current_time
            else:
                cv2.putText(frame_with_results, f"Unknown: {confidence:.2f}%", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame_with_results, "Recognizer not loaded", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.rectangle(frame_with_results, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Reset detection times if face not seen for more than 5 seconds
    current_time = time.time()
    for name in list(face_last_seen.keys()):
        if current_time - face_last_seen[name] > 5:
            detection_times.pop(name, None)
            face_last_seen.pop(name, None)

    return frame_with_results

# Frame generation function
def generate_frames(email):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    start_time = time.time()
    frame_count = 0
    detection_times = {}
    last_detection_time = {}
    face_last_seen = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_with_faces_and_poses = detect_faces_and_poses(frame, detection_times, last_detection_time, face_last_seen, email)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame_with_faces_and_poses, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_with_faces_and_poses)
        frame_with_faces_and_poses = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_faces_and_poses + b'\r\n')

    cap.release()

# Function to continuously generate frames in a separate thread
def run_camera(email):
    while True:
        generate_frames(email)

if __name__ == "__main__":
    # Start the camera in a separate thread
    email = session.get('email')  # Set the email address here
    camera_thread = threading.Thread(target=run_camera, args=(email,))
    camera_thread.daemon = True
    camera_thread.start()