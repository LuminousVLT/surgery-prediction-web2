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
from datetime import datetime

predictor = SurgeryPredictor()
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

            if primary_code in seen_primary_codes: continue
            
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
            
    except Exception as e:
        pass
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
    if request.method != 'POST': return redirect('predict_page')

    try:
        raw_codes = request.POST.getlist('TreatmentCode')
        normalized_codes = list(set([get_primary_code(c) for c in raw_codes])) 
        
        doc_id = request.POST.get('Doctor')
        spec_id = request.POST.get('Specialty')
        complexity_factor = float(request.POST.get('Complexity', 1.0))
        
        room_no = request.POST.get('FacilityRmsNo', 'Unknown')
        case_type = request.POST.get('ORCaseType', 'Elective')
        classified_type = request.POST.get('ORClassifiedType', 'Major')
        
        full_spec_name = next((s['name'] for s in DROPDOWN_DATA['all_specialties'] if s['id'] == spec_id), spec_id)
        
        treatment_names_display = []
        for code in normalized_codes:
            match = next((t for t in DROPDOWN_DATA['treatments'] if t['id'] == code), None)
            treatment_names_display.append(match['text'] if match else f"[{code}] (Mapped)")

        doc_name_full = next((d['text'] for d in DROPDOWN_DATA['doctors'] if d['id'] == doc_id), doc_id)

        # ⭐️ จัดการ DateTime ⭐️
        start_datetime_str = request.POST.get('StartTime', '')
        try:
            if 'T' in start_datetime_str:
                start_hour = int(start_datetime_str.split('T')[1].split(':')[0])
            else:
                start_hour = 9
        except:
            start_hour = 9 

        weight = float(request.POST.get('BodyWeight', 60))
        height = float(request.POST.get('Height', 160))
        age = float(request.POST.get('Age', 40))
        gender = request.POST.get('Gender', 'Unknown')
        anes_type = request.POST.get('AnesthesiaType', 'ANES_GA')
        bmi_val = weight / ((height / 100) ** 2) if height > 0 else 0

        input_data = {
            'Age': age,
            'Height': height,
            'BodyWeight': weight,
            'Gender': gender,
            'Doctor': doc_name_full, 
            'TreatmentCode': normalized_codes, 
            'Specialty': spec_id, 
            'AnesthesiaType': anes_type,
            'ORCaseType': case_type,
            'FacilityRmsNo': room_no,
            'ORClassifiedType': classified_type,
            'Start_Hour': start_hour, 
        }

        result = predictor.predict(input_data)
        base_avg = result.get('avg', 0)
        base_min = result.get('min', 0)
        base_max = result.get('max', 0)
        
        # ปัจจัยความยาก (Complexity Factor)
        final_avg = int(base_avg * complexity_factor)
        final_min = int(base_min * complexity_factor)
        final_max = int(base_max * complexity_factor)
        
        context = {
            'stats': {
                'min': max(5, final_min), 
                'avg': final_avg,
                'max': final_max           
            },
            'doctor_name': doc_name_full,
            'treatment_list': treatment_names_display,
            'main_specialty': full_spec_name,
            'treatment_count': len(normalized_codes),
            'patient_info': {
                'age': int(age),
                'bmi': round(bmi_val, 1),
                'gender': gender,
                'anes': anes_type,
                'room': room_no,
                'case_type': case_type,
                'classified': classified_type
            }
        }
        return render(request, 'result.html', context)

    except Exception as e:
        print(f"❌ Submit Error: {e}")
        return render(request, 'result.html', {'error': str(e)})