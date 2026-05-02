"""
Django settings for surgery_predict project.
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure-fbw$xj!degw%h-j%you*#4)tg@)r3b+tezopkn$hqczxd_#*z#'
DEBUG = True
ALLOWED_HOSTS = ['paniti-jupyter.cra.ac.th', '100.127.9.127', 'localhost', '127.0.0.1']

# --- การตั้งค่าสำหรับ Nginx ที่เจาะจง ---
FORCE_SCRIPT_NAME = '/surgery'
# ให้ STATIC_URL เป็นทางผ่านของ Nginx
STATIC_URL = '/surgery/static/' 
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

CSRF_TRUSTED_ORIGINS = [
    'https://paniti-jupyter.cra.ac.th',
    'http://paniti-jupyter.cra.ac.th',
    'http://100.127.9.127',
    'http://100.127.9.127:6501'
]

# ย้าย WhiteNoise มาควบคุมเรื่องพอร์ตให้แม่นยำขึ้น
WHITENOISE_STATIC_PREFIX = '/static/' 

LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'predict_page'
LOGOUT_REDIRECT_URL = 'login'

INSTALLED_APPS = [
    'jazzmin', # ธีมต้องมาก่อน admin
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware', # ตัวส่งไฟล์ CSS
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'surgery_predict.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'main', 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'surgery_predict.wsgi.application'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

JAZZMIN_SETTINGS = {
    "site_title": "Surgery Predictor Admin",
    "site_header": "Surgery Predictor",
    "site_brand": "Surgery Admin",
    "welcome_sign": "ระบบจัดการหลังบ้าน Surgery Predictor",
    "copyright": "Surgery Predictor V2",
    "show_ui_builder": False,
    "user_avatar": None,
}
JAZZMIN_UI_TWEAKS = {
    "theme": "flatly",
    "dark_mode_theme": None,
}
