from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, send_from_directory
from flask_bcrypt import Bcrypt
from script.camera import generate_frames, reload_email
from script.register import register_camera
from script.captureState import set_capture_complete, is_capture_complete, set_training_complete, is_training_complete
import threading
import secrets
import os
import sqlite3
import shutil
import datetime
import webview

app = Flask(__name__)
app.secret_key = secrets.token_hex()  # Set a secret key for session management

bcrypt = Bcrypt(app)

lock  = threading.Lock()

# Define a global variable to hold the camera thread
camera_thread = None

DATABASE_DIR = 'database'
DATABASE_FILE = 'database.db'
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_FILE)

# Check if the database directory exists, and create it if not
# Ensure directory exists
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)
if not os.path.exists('static/video'):
    os.makedirs('static/video')
if not os.path.exists('static/images'):
    os.makedirs('static/images')
if not os.path.exists('static/detected_faces'):
    os.makedirs('static/detected_faces')
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')
if not os.path.exists('thumbnail'):
    os.makedirs('thumbnail')
if not os.path.exists('logs'):
    os.makedirs('logs')

# Check if the database file exists, and create it if not
if not os.path.exists(DATABASE_PATH):
    print("Database file does not exist. Creating...")
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Create tables or initialize any necessary database setup here
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT NOT NULL,
                        password TEXT NOT NULL
                        )''')
        print("Users table created.")
        conn.execute('''CREATE TABLE IF NOT EXISTS faces (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL
                          )''')
        print("Faces table created.")
        conn.execute('''CREATE TABLE IF NOT EXISTS settings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            source TEXT NOT NULL
                          )''')
        print("Settings table created.")
        # Insert default user if not already present
        cursor = conn.cursor()
        hashed_password = bcrypt.generate_password_hash('admin').decode('utf-8')
        cursor.execute('''INSERT INTO users (email, password) 
                          VALUES (?, ?)''', ('admin@admin', hashed_password))
        cursor.execute('''INSERT INTO settings (source) 
                          VALUES (?)''', (0,))
        conn.commit()
        

# Read the contents of faces
def read_label_to_names():
    label_to_names = []
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM faces")
        rows = cursor.fetchall()
        for row in rows:
            label_to_names.append({"label": row[0], "name": row[1]})
    return label_to_names

# Function to start the camera thread
def start_camera_thread():
    global camera_thread
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=generate_frames)
        camera_thread.daemon = True
        camera_thread.start()

@app.route('/')
def index():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html')

@app.route('/login', methods=['GET','POST'])
def login():
    error = None

    # Check if the user is already authenticated
    if 'authenticated' in session and session['authenticated']:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate login credentials
        try:
            if validate_login(email, password):
                # Set the user as authenticated in the session
                session['authenticated'] = True
                session['email'] = email
                error = None
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid email or password"
        except Exception as e:
            # Log the exception for debugging purposes
            print(f"Error during login: {e}")
            error = "An error occurred during login. Please try again."

    return render_template('login.html', error=error)
# Validation
def validate_login(email, password):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

    if user and bcrypt.check_password_hash(user[0], password):
        return True
    return False

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        # Read the logs
        logsRead = read_logs()
        return render_template('dashboard.html', logs=logsRead)
    else:
        return redirect(url_for('login'))
    
def read_logs():
    logs = []
    current_date = datetime.datetime.now().strftime('%d-%m-%Y')
    logs_file = f'logs/{current_date}.log'
    try:
        with open(logs_file, 'r') as file:
            for line in file:
                # Split the log entry by ' - ' to separate timestamp and message
                log_parts = line.strip().split(' - ')
                if len(log_parts) == 4:
                    timestamp, title, message, media_filename = log_parts
                    log_entry = {'timestamp': timestamp, 'title': title, 'message': message}
                    if media_filename.endswith('.mp4'):
                        log_entry['video_filename'] = media_filename
                    elif media_filename.endswith(('.jpg', '.jpeg', '.png')):
                        log_entry['image_filename'] = media_filename
                    logs.append(log_entry)
                else:
                    logs.append({'timestamp': '', 'title': '', 'message': '', 'video_filename': line.strip()})
    except FileNotFoundError:
        # Handle file not found error
        pass
    return logs
    
@app.route('/video_feed')
def video_feed():
    start_camera_thread()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/history_page')
def history_page():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        # Get the list of all log files in the logs folder
        log_files = os.listdir('logs')
        # Sort log files by modification time
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join('logs', x)), reverse=True)
        # Render the history page with the list of log files
        return render_template('history.html', log_files=log_files)
    else:
        return redirect(url_for('login'))
    
@app.route('/get_log_content')
def get_log_content():
    filename = request.args.get('filename')
    log_path = os.path.join('logs', filename)
    try:
        with open(log_path, 'r') as file:
            content = file.readlines()
            formatted_content = ""
            for line in content:
                # Check if the line contains a video or image filename
                if ".mp4" in line:
                    video_filename = line.split()[-1]
                    formatted_content += f'{line.strip()} <button class="view-media-btn btn btn-outline-info btn-sm" data-media-filename="{video_filename}" data-media-type="video">View Video</button><br>'
                elif any(ext in line for ext in [".jpg", ".jpeg", ".png"]):
                    image_filename = line.split()[-1]
                    formatted_content += f'{line.strip()} <button class="view-media-btn btn btn-outline-info btn-sm" data-media-filename="{image_filename}" data-media-type="image">View Image</button><br>'
                else:
                    formatted_content += line + "<br>"
            return formatted_content
    except FileNotFoundError:
        return 'Log file not found'

@app.route('/view_log/<filename>')
def view_log(filename):
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        log_path = os.path.join('logs', filename)
        try:
            with open(log_path, 'r') as file:
                content = file.read()
            return render_template('view_log.html', content=content)
        except FileNotFoundError:
            return 'Log file not found'
    else:
        return redirect(url_for('login'))

@app.route('/settings')
def settings():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        return render_template('settings.html')
    else:
        return redirect(url_for('login'))

@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        if request.method == 'POST':
            name = request.json.get('name')
            if name:
                session['name'] = name
                return jsonify(success=True)
            else:
                return jsonify(success=False), 400
        return render_template('register_face.html')
    else:
        return redirect(url_for('login'))

@app.route('/register_feed')
def register_feed():
    # Retrieve the name from the session and pass it to the register_camera function
    if 'authenticated' in session and session['authenticated']:
        name = session.get('name')
        if name:
            set_capture_complete(False)  # Reset capture status at the start
            set_training_complete(False)  # Reset training status at the start
            return Response(register_camera(name), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return redirect(url_for('register_face'))
    else:
        return redirect(url_for('login'))

@app.route('/edit_face')
def edit_face():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        return render_template('edit_face.html')
    else:
        return redirect(url_for('login'))

@app.route('/manage_face')
def manage_face():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        label_to_names = read_label_to_names()
        return render_template('manage_face.html', label_to_names=label_to_names)
    else:
        return redirect(url_for('login'))

@app.route('/settings_source', methods=['POST'])
def settings_source():
    if 'authenticated' in session and session['authenticated']:
        data = request.json
        video_source_id = data.get('video_source_id')
        if video_source_id:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE settings SET source = ? WHERE id = 1", (video_source_id,))
                conn.commit()
            return jsonify(success=True)
        else:
            return jsonify(success=False), 400
    else:
        return jsonify(success=False), 403

@app.route('/help_page')
def help_page():
    return render_template('help.html')
    
@app.route('/user_profile')
def user_profile():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        email = session.get("email")
        return render_template('user_profile.html', email=email)
    else:
        return redirect(url_for('login'))
    
@app.route('/change_email', methods=['POST'])
def change_email():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        new_email = request.form['newEmail']
        confirm_email = request.form['confirmEmail']
        password = request.form['password']
        
        # Check if new email and confirm email match
        if new_email != confirm_email:
            return "New email and confirm email do not match.", 400
        
        # Validate password
        email = session['email']  # Get the current email from session or wherever it's stored
        if not validate_login(email, password):
            return "Incorrect password.", 400
        
        # Update email in the database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET email = ? WHERE email = ?", (new_email, email))
            conn.commit()
            
        # Update the email in the session
        session['email'] = new_email

        reload_email()

        return "Email updated successfully."
    else:
        return redirect(url_for('login'))

@app.route('/change_password', methods=['POST'])
def change_password():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        current_password = request.form['currentPassword']
        new_password = request.form['newPassword']
        confirm_new_password = request.form['confirmNewPassword']
        
        # Check if new password and confirm password match
        if new_password != confirm_new_password:
            return "Passwords do not match.", 400
        
        # Validate current password
        email = session['email']  # Get the current email from session or wherever it's stored
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
        
        if user and bcrypt.check_password_hash(user[0], current_password):
            # Hash the new password
            hashed_new_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
            
            # Update password in the database
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_new_password, email))
                conn.commit()
            
            return "Password updated successfully."
        else:
            return "Incorrect current password.", 400
    else:
        return redirect(url_for('login'))
    
@app.route('/check_completion')
def check_completion():
    return jsonify({"complete": is_capture_complete()})

@app.route('/check_training_completion')
def check_training_completion():
    return jsonify({"complete": is_training_complete()})

@app.route('/delete_entry', methods=['POST'])
def delete_entry():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        data = request.json
        entry_id = data.get('label')

        if entry_id:
            # Connect to the database
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                # Delete the entry from the faces table
                cursor.execute("DELETE FROM faces WHERE id = ?", (entry_id,))
                conn.commit()

            # Delete the folder associated with the entry_id
            folder_path = os.path.join("captured_images", str(entry_id))
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            return jsonify(success=True)
        else:
            return jsonify(success=False), 400
    else:
        return jsonify(success=False), 403

@app.route('/video/<filename>')
def video(filename):
    return send_from_directory('static/video', filename)

@app.route('/media/<media_type>/<filename>')
def media(media_type, filename):
    if media_type == 'video':
        return send_from_directory('static/video', filename)
    elif media_type == 'image':
        return send_from_directory('static/images', filename)
    else:
        return 'Invalid media type.', 400

@app.route('/get_activity_history')
def get_activity_history():
    # Read the logs
    logsRead = read_logs()
    # Render the activity history section
    return render_template('activity_history.html', logs=logsRead)
    
webview.create_window('Lambda', app)

if __name__ == '__main__':
    #app.run(debug=True, threaded=True)
    webview.start()