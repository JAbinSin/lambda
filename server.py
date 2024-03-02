from flask import Flask, render_template, Response, request, redirect, url_for, session
from camera import generate_frames
import threading

app = Flask(__name__)
app.secret_key = 'L1Mb1Q1FRI6b'  # Set a secret key for session management

lock  = threading.Lock()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET','POST'])
def login():
    error = None

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate login credentials
        try:
            if validate_login(email, password):
                # Set the user as authenticated in the session
                session['authenticated'] = True
                error = None
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid email or password"
        except Exception as e:
            # Log the exception for debugging purposes
            print(f"Error during login: {e}")
            error = "An error occurred during login. Please try again."

    return render_template('login.html', error=error)

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    # Check if the user is authenticated
    if 'authenticated' in session and session['authenticated']:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Validation
def validate_login(email, password):
    # Check if the provided credentials match the predefined values
    return email == 'admin@admin' and password == 'admin'

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
