#!/bin/bash

source .venv/bin/activate  # Activate your virtual environment if you have one
export FLASK_APP=server.py
export FLASK_RUN_PORT=5000  # Set your desired port

# Create a logs directory if it doesn't exist
mkdir -p logs

# Run your Flask application with --host=0.0.0.0
flask run --host=0.0.0.0 --debug > logs/system.log 2>&1
