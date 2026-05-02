#!/bin/bash
cd /home/hds/Webapp/SurgeryPredictor
source venv/bin/activate

echo "Stopping existing processes..."
pkill -u hds -f "waitress-serve"

echo "Force cleaning and collecting static files (Jazzmin Fix)..."
# ลบโฟลเดอร์ staticfiles เดิมทิ้งเพื่อให้มั่นใจว่าไม่มีของเก่าค้าง
rm -rf staticfiles
python manage.py collectstatic --noinput

echo "Starting server on port 6501..."
nohup waitress-serve --port=6501 surgery_predict.wsgi:application > output.log 2>&1 &

sleep 2
if ps aux | grep -v grep | grep "waitress-serve" > /dev/null
then
    echo "SUCCESS: Server is running on port 6501"
else
    echo "ERROR: Server failed to start! Check collectstatic.log or output.log"
fi
