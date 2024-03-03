import cv2
import time
from ultralytics import YOLO

# Load the pre-trained YOLOv8 pose estimation model
pose_model = YOLO("yolov8n-pose.pt")

# Load the pre-trained face cascade outside of the function
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Run YOLOv8 interface on the frame
    results = pose_model.predict(frame, show_labels=False, show_conf=False, show_boxes=False)

    # Visualize the results on the frame
    frame = results[0].plot()
    
    return frame

def generate_frames():
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
    
    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Resize the frame for better performance
        frame = cv2.resize(frame, (640, 480))
        
        # Perform face detection
        frame_with_faces = detect_faces(frame)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame_with_faces, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_faces)
        frame_with_faces = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_faces + b'\r\n')

    # Release the camera
    cap.release()

if __name__ == "__main__":
    generate_frames()
