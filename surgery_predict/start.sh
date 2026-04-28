#!/bin/bash
# สคริปต์เวอร์ชัน "เปิดทุกประตู" เพื่อแก้ปัญหา 502 Bad Gateway
cd /home/hds/Webapp/SurgeryPredictor

echo "Cleaning up..."
pkill -f "waitress-serve"
sleep 2

source venv/bin/activate

echo "Starting server on [::]:6501 (Both IPv4 and IPv6)..."
# ใช้ --host=0.0.0.0 เพื่อให้รับได้ทั้ง 127.0.0.1 และไอพีเครื่อง
# หรือใช้ --host=[::] ถ้าเซิร์ฟเวอร์เน้น IPv6
nohup waitress-serve --host=0.0.0.0 --port=6501 surgery_predict.wsgi:application > output.log 2>&1 &

echo "Done! Program is listening on port 6501"
