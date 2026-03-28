# 1. ปรับมาใช้ Python 3.13 (เวอร์ชันเดียวกับเครื่องคุณเจมส์)
FROM python:3.13-slim

# 2. ตั้งค่าระบบ
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. กำหนดโฟลเดอร์ทำงาน
WORKDIR /app

# 4. ⭐️ ลงไลบรารีระบบ (เพิ่ม build-essential สำหรับ Python 3.13)
# เนื่องจาก 3.13 ใหม่มาก บาง Wheel อาจจะยังไม่มี ต้องใช้ gcc ช่วยสร้างครับ
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. อัปเกรดเครื่องมือติดตั้ง
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel

# 6. ติดตั้งไลบรารี (เพิ่ม --use-pep517 ในกรณีเจอ Error ตอน Build)
RUN pip install --no-cache-dir -r requirements.txt

# 7. ก๊อปปี้ไฟล์โปรเจกต์
COPY . /app/

# 8. รัน Server
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]