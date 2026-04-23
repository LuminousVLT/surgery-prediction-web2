from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
import pandas as pd
import numpy as np
import os
from .feature_engineer import SurgeryPredictor
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth import logout as auth_logout 

# สร้าง Instance ของโมเดลไว้ให้พร้อมใช้งาน
predictor = SurgeryPredictor()

# ดึง Audit Map ล่าสุดที่เพิ่งเทรนเสร็จมาจาก predictor
TREATMENT_MAPPING = getattr(predictor, 'audit_map', {})

def get_primary_code(code):
    clean_code = str(code).strip().upper()
    return TREATMENT_MAPPING.get(clean_code, clean_code)

def get_dropdown_data():
    data_path = os.path.join(settings.BASE_DIR, 'data')
    context = {'doctors': [], 'treatments': [], 'all_specialties': []}
    
    try:
        doc_df = pd.read_excel(os.path.join(data_path, 'DoctorName.xlsx')).fillna('')
        spec_file = os.path.join(data_path, 'Treatment_Specialty.xlsx')
        treat_df = pd.read_excel(spec_file, sheet_name='Query').fillna('')
        
        try:
            choice_df = pd.read_excel(spec_file, sheet_name='Choice').fillna('')
            spec_full_names = dict(zip(choice_df['Type'].astype(str), choice_df['Name'].astype(str)))
        except:
            spec_full_names = {}

        for _, row in doc_df.iterrows():
            d_id = str(row.get('Doctor', '')).strip()
            if d_id:
                context['doctors'].append({
                    'id': d_id, 
                    'text': f"[{d_id}] {row.get('DoctorName', '')}",
                    'spec': str(row.get('Specialty', 'Surgery'))
                })

        all_specs_raw = sorted(treat_df['SpecialtyName'].unique().tolist())
        for s in all_specs_raw:
            s_str = str(s).strip()
            if s_str.lower() != 'anes' and s_str != '' and s_str.lower() != 'nan':
                full_name = spec_full_names.get(s_str, s_str)
                context['all_specialties'].append({'id': s_str, 'name': full_name})

        seen_primary_codes = set()
        for _, row in treat_df.iterrows():
            raw_code = str(row.get('TreatmentCode', '')).strip().upper()
            spec_name = str(row.get('SpecialtyName', '')).strip()
            
            primary_code = get_primary_code(raw_code)

            if primary_code in seen_primary_codes:
                continue
            
            name = str(row.get('TreatmentName', '')).strip()
            if not name or name == 'nan':
                name = str(row.get('TreatmentLocalName', '')).strip()
            
            if spec_name.lower() == 'anes' or raw_code.startswith('PF'): continue
            
            context['treatments'].append({
                'id': primary_code,
                'text': f"[{primary_code}] {name}",
                'spec': spec_name
            })
            
            seen_primary_codes.add(primary_code)
            
        print(f"✅ [System] สร้าง Dropdown สำเร็จ: หัตถการ {len(context['treatments'])} รายการ")
            
    except Exception as e:
        print(f"❌ [Error] สร้าง Dropdown ล้มเหลว: {e}")
    return context

DROPDOWN_DATA = get_dropdown_data()

def root_redirect(request):
    return redirect('login')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'ลงทะเบียนและเข้าสู่ระบบสำเร็จ!')
            return redirect('predict_page') 
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    auth_logout(request)
    messages.success(request, 'ออกจากระบบเรียบร้อยแล้ว')
    return redirect('login')

@login_required
def predict_page(request):
    return render(request, 'predict.html', DROPDOWN_DATA)

@login_required
def predict_submit(request):
    if request.method != 'POST':
        return redirect('predict_page')

    try:
        raw_codes = request.POST.getlist('TreatmentCode')
        normalized_codes = list(set([get_primary_code(c) for c in raw_codes])) 
        
        doc_id = request.POST.get('Doctor')
        spec_id = request.POST.get('Specialty')
        complexity_factor = float(request.POST.get('Complexity', 1.0))
        
        full_spec_name = next((s['name'] for s in DROPDOWN_DATA['all_specialties'] if s['id'] == spec_id), spec_id)
        
        treatment_names_display = []
        for code in normalized_codes:
            match = next((t for t in DROPDOWN_DATA['treatments'] if t['id'] == code), None)
            treatment_names_display.append(match['text'] if match else f"[{code}] (Mapped)")

        doc_name_full = next((d['text'] for d in DROPDOWN_DATA['doctors'] if d['id'] == doc_id), doc_id)

        start_time_str = request.POST.get('StartTime', '09:00')
        try:
            start_hour = int(start_time_str.split(':')[0])
        except:
            start_hour = 9 

        input_data = {
            'Age': float(request.POST.get('Age', 40)),
            'Height': float(request.POST.get('Height', 160)),
            'BodyWeight': float(request.POST.get('BodyWeight', 60)),
            'Gender': request.POST.get('Gender', 'Unknown'),
            'Doctor': doc_name_full, 
            'TreatmentCode': normalized_codes, 
            'Specialty': spec_id, 
            'AnesthesiaType': request.POST.get('AnesthesiaType', 'ANES_GA'),
            'ORCaseType': request.POST.get('ORCaseType', 'Unknown'),
            'FacilityRmsNo': request.POST.get('FacilityRmsNo', 'Unknown'),
            'ORClassifiedType': request.POST.get('ORClassifiedType', 'Unknown'),
            'Start_Hour': start_hour, 
        }

        # ทำนายผลด้วย AI (Optimized Weighted Ensemble)
        result = predictor.predict(input_data)
        base_time = result['minutes']
        final_time = int(base_time * complexity_factor)

        # ⭐️ อัปเดต MAE_DICT จากผลการทดสอบล่าสุด (Auto-Tuned)
        MAE_DICT = {
            'จักษุวิทยา (Ophthalmology)': 9,
            'ศัลยกรรมระบบทางเดินอาหาร (Gastrointestinal Surgery)': 12,
            'ตจวิทยา/ผิวหนัง (Dermatology)': 16,
            'ศัลยกรรมทางเดินปัสสาวะ (Urology)': 17,
            'ศัลยกรรมเต้านม (Breast Surgery)': 22,
            'ศัลยกรรมกระดูกและข้อ (Orthopedics)': 24,
            'นรีเวชวิทยา (Gynecology)': 25,
            'โสต ศอ นาสิกวิทยา (Otolaryngology / ENT)': 28,
            'ศัลยกรรมหัวใจและทรวงอก (Cardiovascular and Thoracic Surgery)': 34,
            'ศัลยกรรมทั่วไป (General Surgery)': 40
        }

        # ดึงค่า MAE ของแผนกนั้นๆ มาใช้ ถ้าไม่ตรงกับในดิก ให้ใช้ 20 เป็นค่า Default
        dept_mae = MAE_DICT.get(full_spec_name, 20)

        model_details = result.get('details', {})
        chart_data = {
            'xgb': int(model_details.get('XGBoost', base_time) * complexity_factor),
            'lgb': int(model_details.get('LightGBM', base_time) * complexity_factor),
            'cat': int(model_details.get('CatBoost', base_time) * complexity_factor)
        }
        
        context = {
            'stats': {
                'min': max(15, int(final_time - dept_mae)), # บังคับขั้นต่ำ 15 นาที
                'avg': final_time,
                'max': int(final_time + dept_mae)           # บวกด้วย MAE ของแผนก
            },
            'doctor_name': doc_name_full,
            'treatment_list': treatment_names_display,
            'main_specialty': full_spec_name,
            'treatment_count': len(normalized_codes),
            'model_details': chart_data
        }
        return render(request, 'result.html', context)

    except Exception as e:
        print(f"❌ Submit Error: {e}")
        return render(request, 'result.html', {'error': str(e)})