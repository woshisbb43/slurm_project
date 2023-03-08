#!/bin/bash

# Stop the Flask app
pkill -f "python3 app.py"

# Wait for the app to shut down
sleep 5

# Start the Flask app
cd /data/web/flask_upload
python3 app.py &

