import cv2
import time
import logging
import sqlite3
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from ultralytics import YOLO
import threading
import xgboost as xgb
import pandas as pd
import joblib
from flask import session
from fer import FER  # Importing FER library

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
formatter = logging.Formatter('%(asctime)s - %(message)s ')
file_handler.setFormatter(formatter)

# Add file handler to logger
detection_logger.addHandler(file_handler)

# Load the pre-trained YOLOv8 pose estimation model
pose_model = YOLO("yolov8n-pose.pt")

# Load the XGBoost model
model_xgb = xgb.Booster()
model_xgb.load_model('trained_models/model_weights.xgb')
scaler = joblib.load('trained_models/scaler.pkl')

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define keypoint labels for the pose model
labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Initialize the FER emotion detection model
emotion_detector = FER()

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
import threading
import time

def send_email_notification(name, subject, message, email, attachment_path=None):
    def send_email():

        # Create a multipart message
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = 'lambda54312@gmail.com'
        msg['To'] = email

        # Attach the body message
        body = MIMEText(message, 'plain')
        msg.attach(body)

        # Attach a file if provided
        if attachment_path:
            try:
                with open(attachment_path, 'rb') as attachment:
                    part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                    msg.attach(part)
            except Exception as e:
                print(f"Failed to attach file: {e}")

        # Send the email
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login('lambda54312@gmail.com', 'yowc qcuc dgxg ngol')
                server.send_message(msg)
            print(f"Email sent to {email}")
        except Exception as e:
            print(f"Failed to send email: {e}")

    email_thread = threading.Thread(target=send_email)
    email_thread.start()

# Face detection and pose estimation function
def detect_faces_and_poses(frame, detection_times, last_detection_time, face_last_seen, email, angry_detection_times, action_detection_times, video_writer, recording, pending_recording):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    results = pose_model(frame, verbose=False)
    frame_with_results = frame.copy()
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_file = None
    save_thumb = False

    # Check if any keypoints are detected
    if results and results[0].keypoints is not None and results[0].keypoints.conf is not None:
        keypoints_numpy = results[0].keypoints.xyn.cpu().numpy()[0]
        keypoints_scores = results[0].keypoints.conf.cpu().numpy()[0]

        # Threshold for keypoint confidence
        confidence_threshold = 0.5
        labeled_keypoints = [[labels[i], kp[0], kp[1], score] for i, (kp, score) in enumerate(zip(keypoints_numpy, keypoints_scores))]
        visible_keypoints = [label for label, x, y, score in labeled_keypoints if 0 <= x <= 1 and 0 <= y <= 1 and score > confidence_threshold]
    else:
        visible_keypoints = []

    current_time = time.time()  # Ensure current_time is defined at the beginning of the function

    for r in results:
        bound_box = r.boxes.xyxy
        conf = r.boxes.conf.tolist()
        keypoints = r.keypoints.xyn.tolist()

        for index, box in enumerate(bound_box):
            if conf[index] > 0.80:
                data = {}

                # Initialize the x and y lists for each possible key
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]

                df = pd.DataFrame(data, index=[0])
                df_scaled = scaler.transform(df)
                dmatrix = xgb.DMatrix(df_scaled)
                behavior = model_xgb.predict(dmatrix)
                binary_predictions = int(behavior[0])  # Get the class with the highest probability

                # Determine action label
                action_labels = ['squatting', 'standing', 'punching', 'kicking']
                action_label = action_labels[binary_predictions]

                # Check the visibility of required keypoints
                kicking_keypoints_visible = (
                    all(kp in visible_keypoints for kp in ["right_hip", "right_knee", "right_ankle"]) or
                    all(kp in visible_keypoints for kp in ["left_hip", "left_knee", "left_ankle"])
                )
                
                punching_keypoints_visible = (
                    all(kp in visible_keypoints for kp in ["right_shoulder", "right_elbow", "right_wrist"]) or
                    all(kp in visible_keypoints for kp in ["left_shoulder", "left_elbow", "left_wrist"])
                )

                x, y, w, h = 0, 0, 0, 0  # Initialize x, y, w, h with default values
                if action_label == "kicking" and kicking_keypoints_visible:
                    cv2.putText(frame_with_results, action_label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    if (x, y, w, h) not in pending_recording:
                        pending_recording[(x, y, w, h)] = current_time
                    else:
                        if current_time - pending_recording[(x, y, w, h)] >= 1.5:
                            action_detection_times[(x, y, w, h)] = current_time
                            if not recording:
                                video_file = datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
                                video_filename = os.path.join('static', 'video', video_file)
                                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
                                recording = True

                                if save_thumb == False:
                                    video_image_path = f"thumbnail/{int(current_time)}.png"
                                    cv2.imwrite(video_image_path, frame_with_results)
                                    save_thumb = True
                            if video_file:
                                detection_logger.info(f"Person detected kicking - {video_file}")
                                subject = "Aggressive Behavior Detected"
                                message = "Person detected showing aggressive behavior - Kicking"
                                send_email_notification('', subject, message, email, video_image_path)
                                save_thumb = False
                elif action_label == "punching" and punching_keypoints_visible:
                    cv2.putText(frame_with_results, action_label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    if (x, y, w, h) not in pending_recording:
                        pending_recording[(x, y, w, h)] = current_time
                    else:
                        if current_time - pending_recording[(x, y, w, h)] >= 1.5:
                            action_detection_times[(x, y, w, h)] = current_time
                            if not recording:
                                video_file = datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
                                video_filename = os.path.join('static', 'video', video_file)
                                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
                                recording = True

                                if save_thumb == False:
                                    video_image_path = f"thumbnail/{int(current_time)}.png"
                                    cv2.imwrite(video_image_path, frame_with_results)
                                    save_thumb = True
                            if video_file:
                                detection_logger.info(f"Person detected punching - {video_file}")
                                subject = "Aggressive Behavior Detected"
                                message = "Person detected showing aggressive behavior - Punching"
                                send_email_notification('', subject, message, email, video_image_path)
                                save_thumb = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        try:
            result = emotion_detector.detect_emotions(roi_color)
            if result:
                emotion_confidences = result[0]["emotions"]
                dominant_emotion = max(emotion_confidences, key=emotion_confidences.get)
                print(dominant_emotion, emotion_confidences[dominant_emotion])

                if dominant_emotion == "angry":
                    if emotion_confidences[dominant_emotion] > 0.40:  # Adjust threshold for confidence
                        if (x, y, w, h) not in pending_recording:
                            pending_recording[(x, y, w, h)] = current_time
                        else:
                            if current_time - pending_recording[(x, y, w, h)] >= 1:
                                angry_detection_times[(x, y, w, h)] = current_time
                                if not recording:
                                    video_file = datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
                                    video_filename = os.path.join('static', 'video', video_file)
                                    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
                                    recording = True
                                    if save_thumb == False:
                                        video_image_path = f"thumbnail/{int(current_time)}.png"
                                        cv2.imwrite(video_image_path, roi_color)
                                        save_thumb = True
                                if video_file:
                                    detection_logger.info(f"Angry person detected - {video_file}")
                                    subject = "Emotional Instability Detected"
                                    message = "Angry person detected"
                                    send_email_notification('', subject, message, email, video_image_path)
                                    save_thumb = False
                    else:
                        if (x, y, w, h) in pending_recording:
                            pending_recording.pop((x, y, w, h))
                else:
                    if (x, y, w, h) in pending_recording:
                        pending_recording.pop((x, y, w, h))

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

            if face_name and confidence > 0.05:
                name = face_name[0]
                cv2.putText(frame_with_results, f"{name}: {confidence:.2f}%", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if name not in detection_times:
                    detection_times[name] = current_time
                else:
                    if current_time - detection_times[name] >= 5 and (current_time - last_detection_time.get(name, 0)) >= 5:
                        face_image_path = f"detected_faces/{name}_{int(current_time)}.png"
                        cv2.imwrite(face_image_path, roi_color)

                        subject = "Lambda System"
                        message = "Face Detected from " + name
                        send_email_notification(name, subject, message, email, face_image_path)
                        last_detection_time[name] = current_time
                face_last_seen[name] = current_time
            else:
                cv2.putText(frame_with_results, f"Unknown: {confidence:.2f}%", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame_with_results, "Recognizer not loaded", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.rectangle(frame_with_results, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Stop recording if no actions are detected for more than 5 seconds
    if recording:
        if all(current_time - t > 5 for t in angry_detection_times.values()) and all(current_time - t > 5 for t in action_detection_times.values()):
            recording = False
            video_writer.release()
            video_writer = None

    if recording:
        video_writer.write(frame)

    return frame_with_results, video_writer, recording, pending_recording

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
    angry_detection_times = {}
    action_detection_times = {}  # Dictionary to track action detection times
    pending_recording = {}  # Dictionary to track pending recordings
    video_writer = None
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_with_faces_and_poses, video_writer, recording, pending_recording = detect_faces_and_poses(
            frame, detection_times, last_detection_time, face_last_seen, email, angry_detection_times, action_detection_times, video_writer, recording, pending_recording
        )

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame_with_faces_and_poses, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_with_faces_and_poses)
        frame_with_faces_and_poses = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_faces_and_poses + b'\r\n')

    if recording:
        video_writer.release()

    cap.release()

if __name__ == "__main__":
    # Start the camera in a separate thread
    email = session.get('email')  # Set the email address here
    camera_thread = threading.Thread(target=generate_frames, args=(email,))
    camera_thread.daemon = True
    camera_thread.start()
