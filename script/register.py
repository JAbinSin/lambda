import cv2
import os
import numpy as np
import sqlite3
from flask import session
from script.captureState import set_capture_complete, set_training_complete
from script.camera import reload_lbph_trained_model

def register_camera(name):
    # Connect to SQLite3 database using a context manager
    with sqlite3.connect(f'database/database.db') as conn:
        cursor = conn.cursor()

        # Get the latest label (maximum value) from the 'faces' table
        cursor.execute("SELECT seq FROM sqlite_sequence WHERE name = 'faces'")
        seq = cursor.fetchone()
        label = seq[0] + 1 if seq is not None else 1

        # Insert name into the 'faces' table
        cursor.execute("INSERT INTO faces (name) VALUES (?)", (name,))
        conn.commit()

    # Initialize the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the camera (0 represents the default camera)
    cap = cv2.VideoCapture(0)

    # Set the resolution to 720p
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    # Set the frame rate to 60fps
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create a folder with the label to save the captured images if it doesn't exist
    output_folder = os.path.join("captured_images", str(label))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    count = 1
    first_image_saved = False  # Flag to track if the first image is saved

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize the frame for better performance
        frame = cv2.resize(frame, (640, 480))
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the Haar cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Crop the face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Save the face image with the appropriate label as part of the filename
            image_path = os.path.join(output_folder, f"{label}_face_{count}.jpg")
            cv2.imwrite(image_path, face_roi)
            print(f"Image {count} captured.")
            count += 1

            # Save the first image to static/images
            if not first_image_saved:
                first_image_saved = True
                first_image_path = os.path.join("static/images", f"{label}_face_1.jpg")
                print(first_image_path)
                cv2.imwrite(first_image_path, face_roi)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the number of images captured on the frame
            cv2.putText(frame, f"Images captured: {count-1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

        # Break the loop after capturing 100 images
        if count > 300:
            set_capture_complete(True)
            break

    # Release the camera
    cap.release()
    yield b'COMPLETE'

    # Training LBPH recognizer with all captured images
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    faces, labels = [], []

    captured_images_path = "captured_images"
    for person_folder in os.listdir(captured_images_path):
        person_label = int(person_folder)
        person_folder_path = os.path.join(captured_images_path, person_folder)
        if os.path.isdir(person_folder_path):
            for filename in os.listdir(person_folder_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(person_folder_path, filename)
                    face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(face_image)
                    labels.append(person_label)

    if faces and labels:
        # Train the recognizer
        recognizer.train(faces, np.array(labels))
        # Save the trained model
        if not os.path.exists("trained_models"):
            os.makedirs("trained_models")
        recognizer.save(f"trained_models/lbph_trained_model.yml")
        print("LBPH model trained and saved.")
        reload_lbph_trained_model()
    
    set_training_complete(True)

if __name__ == "__main__":
    name = session.get('name')
    register_camera(name)
    reload_lbph_trained_model()
