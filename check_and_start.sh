#!/bin/bash

# Check if the start_flask.sh process is running
if ! pgrep -f "start_flask.sh" > /dev/null; then
    # If not running, start the script
    cd /home/orangepi/lambda && ./start_flask.sh >> /home/orangepi/lambda/logs/cron.log 2>&1
fi
