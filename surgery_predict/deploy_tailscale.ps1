# --- รายละเอียดการตั้งค่า ---
$IP = "100.127.9.127"
$USER = "hds"
$LOCAL_PATH = "C:\Users\damon\Downloads\james_project_surgery\Surgery_Durationn_time_v2-main"
$REMOTE_PATH = "/home/hds/Webapp/SurgeryPredictor"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   🚀 Deploying Surgery Predictor to Server   " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. อัปโหลดไฟล์
Write-Host ">>> 1. กำลังอัปโหลดโค้ดล่าสุดไปยังเซิร์ฟเวอร์..." -ForegroundColor Yellow
# ใช้ scp ส่งไฟล์
scp -r "$LOCAL_PATH\*" "${USER}@${IP}:${REMOTE_PATH}"

# 2. สั่งรันสคริปต์ Start
Write-Host ">>> 2. กำลังสั่งรีสตาร์ทเซิร์ฟเวอร์..." -ForegroundColor Yellow
ssh ${USER}@${IP} "bash ${REMOTE_PATH}/start.sh"

Write-Host "------------------------------------------" -ForegroundColor Green
Write-Host " ✅ DONE! เข้าเว็บได้ที่: http://$IP/surgery/login/" -ForegroundColor Green
Write-Host "------------------------------------------" -ForegroundColor Green
pause