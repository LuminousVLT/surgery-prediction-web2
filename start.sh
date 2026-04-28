#!/bin/bash
cd /home/hds/Webapp/SurgeryPredictor
source venv/bin/activate

echo "Stopping any existing processes on port 6501..."
pkill -u hds -f "waitress-serve --port=6501"

echo "Starting server on 0.0.0.0:6501..."
nohup waitress-serve --port=6501 surgery_predict.wsgi:application > output.log 2>&1 &

sleep 2
if ps aux | grep -v grep | grep "waitress-serve --port=6501" > /dev/null
then
    echo "SUCCESS: Server is running on port 6501"
else
    echo "ERROR: Port 6501 is still BUSY. Wait for Admin to clear it."
fi
