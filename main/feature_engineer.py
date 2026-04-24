import pandas as pd
import numpy as np
import joblib
import os
import re
import lightgbm as lgb
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models')

class SurgeryPredictor:
    def __init__(self):
        self.is_ready = False
        self.model_avg = None
        self.model_min = None
        self.model_max = None
        self.feature_columns = []
        self.load_resources()

    def load_resources(self):
        try:
            model_file = os.path.join(MODEL_PATH, 'surgery_duration_models_100pct.pkl')
            if not os.path.exists(model_file): return

            saved_data = joblib.load(model_file)
            
            # ⭐️ โหลดโมเดลทั้ง 3 ตัวของ LightGBM
            self.model_avg = saved_data.get('lgb_avg')
            self.model_min = saved_data.get('lgb_min')
            self.model_max = saved_data.get('lgb_max')
            
            self.code_weights = saved_data['code_weights']
            self.audit_map = saved_data.get('audit_map', {}) 
            self.doc_stats = saved_data['doc_stats']
            self.spec_stats = saved_data['spec_stats']
            self.global_mean = saved_data['global_mean']
            self.feature_columns = saved_data['feature_names']
            
            ds_df = saved_data['doc_spec_stats']
            self.ds_map = ds_df.set_index(['Doctor', 'Specialty'])['Doc_Spec_Avg'].to_dict()
            
            da_df = saved_data['doc_anes_stats']
            self.da_map = da_df.set_index(['Doctor', 'AnesthesiaType'])['Doc_Anes_Avg'].to_dict()
            
            self.is_ready = True
            print(f"✅ [AI Engine] โหลด Production Model (LightGBM Min/Avg/Max) สำเร็จ!")
            
        except Exception as e:
            print(f"❌ [AI Engine] เกิดข้อผิดพลาดตอนโหลดทรัพยากร: {e}")
            self.is_ready = False

    def preprocess_input(self, input_data):
        if not self.is_ready: return None
        
        age = float(input_data.get('Age', 40.0))
        h = float(input_data.get('Height', 160.0))
        w = float(input_data.get('BodyWeight', 60.0))
        bmi = w / ((h / 100) ** 2) if h > 0 else 22.0
        
        if bmi < 18.5: bmi_cat = 'Under'
        elif bmi < 25: bmi_cat = 'Normal'
        elif bmi < 30: bmi_cat = 'Over'
        else: bmi_cat = 'Obese'

        start_hour = int(input_data.get('Start_Hour', 9))
        day_of_week = int(input_data.get('Day_of_Week', pd.Timestamp.now().dayofweek))
        time_period = 'Morning' if start_hour < 11 else ('Afternoon' if start_hour < 16 else 'Night')
        h_sin = np.sin(2 * np.pi * start_hour / 24)
        h_cos = np.cos(2 * np.pi * start_hour / 24)

        raw_codes = input_data.get('TreatmentCode', [])
        if isinstance(raw_codes, str): raw_codes = [raw_codes]
        
        mapped_codes = [self.audit_map.get(str(c).strip().upper(), str(c).strip().upper()) for c in raw_codes]
        weights = [self.code_weights.get(c, self.global_mean) for c in mapped_codes]
        
        if weights:
            main_comp = max(weights)
            support_comp = sum(weights) - main_comp
            main_code = mapped_codes[weights.index(main_comp)]
        else:
            main_comp = self.global_mean
            support_comp = 0
            main_code = 'Unknown'

        raw_doc = str(input_data.get('Doctor', 'Unknown')).strip()
        match = re.match(r'\[(.*?)\]\s*(.*)', raw_doc)
        doc_format_1 = f"{match.group(2).strip()} ({match.group(1).strip()})" if match else raw_doc
        doc_format_2 = match.group(1).strip() if match else raw_doc 
        
        if doc_format_1 in self.doc_stats: doctor_clean = doc_format_1
        elif doc_format_2 in self.doc_stats: doctor_clean = doc_format_2
        else: doctor_clean = raw_doc

        spec = str(input_data.get('Specialty', 'Unknown'))
        anes = str(input_data.get('AnesthesiaType', 'ANES_GA'))

        spec_avg = self.spec_stats.get(spec, self.global_mean)
        doc_avg = self.doc_stats.get(doctor_clean, spec_avg)
        ds_avg = self.ds_map.get((doctor_clean, spec), doc_avg)
        da_avg = self.da_map.get((doctor_clean, anes), doc_avg)

        df_dict = {
            'Start_Hour': start_hour, 'Day_of_Week': day_of_week, 'Time_Period': time_period,
            'Hour_Sin': h_sin, 'Hour_Cos': h_cos, 'Age': age,
            'FacilityRmsNo': str(input_data.get('FacilityRmsNo', 'Unknown')),
            'ORClassifiedType': str(input_data.get('ORClassifiedType', 'Unknown')),
            'ORCaseType': str(input_data.get('ORCaseType', 'Unknown')),
            'Height': h, 'BodyWeight': w, 'BMIValue': bmi, 'BMI_Cat': bmi_cat,
            'Gender': str(input_data.get('Gender', 'Unknown')),
            'Main_TreatmentCode': main_code, 'Procedure_Count': len(mapped_codes),
            'AnesthesiaType': anes, 'Specialty': spec, 'Doctor': doctor_clean,
            'Main_Complexity': main_comp, 'Support_Complexity': support_comp,
            'Doctor_AvgTime': doc_avg, 'Doc_Spec_Avg': ds_avg, 'Doc_Anes_Avg': da_avg
        }

        final_df = pd.DataFrame([df_dict])
        
        cat_cols = ['Gender', 'FacilityRmsNo', 'ORClassifiedType', 'ORCaseType', 'AnesthesiaType', 'Day_of_Week', 'Doctor', 'Main_TreatmentCode', 'Specialty', 'Time_Period', 'BMI_Cat']
        for c in cat_cols:
            if c in final_df.columns: final_df[c] = final_df[c].astype(str).replace('nan', 'Unknown').astype('category')

        for col in self.feature_columns:
            if col not in final_df.columns:
                if col in cat_cols:
                    final_df[col] = 'Unknown'
                    final_df[col] = final_df[col].astype('category')
                else: final_df[col] = 0.0

        return final_df.reindex(columns=self.feature_columns)

    def predict(self, input_data):
        try:
            X = self.preprocess_input(input_data)
            if X is None: return {'avg': 0, 'min': 0, 'max': 0, 'details': {}}

            # ⭐️ ให้ AI ทั้ง 3 ตัวทาย (ไม่ต้องถอด expm1 เพราะตอนเทรนไม่ได้แปลง Log)
            p_avg = max(self.model_avg.predict(X)[0], 0)
            
            # ป้องกันกรณีโหลดโมเดล Min/Max ไม่ผ่าน
            if self.model_min and self.model_max:
                p_min = max(self.model_min.predict(X)[0], 0)
                p_max = max(self.model_max.predict(X)[0], 0)
            else:
                p_min = p_avg * 0.8
                p_max = p_avg * 1.2
            
            # การ์ดป้องกัน (กรณี Min ดันทายได้มากกว่า Avg ให้สลับ)
            if p_min > p_avg: p_min = p_avg * 0.8
            if p_max < p_avg: p_max = p_avg * 1.2
            
            return {
                'avg': int(p_avg),
                'min': int(p_min),
                'max': int(p_max),
                'details': {'LightGBM': int(p_avg)} # ส่งชื่อไปให้ views วาดกราฟแท่งเดียว
            }
        except Exception as e:
            print(f"❌ [AI Engine] Prediction Error: {e}")
            return {'avg': 0, 'min': 0, 'max': 0, 'details': {'error': str(e)}}