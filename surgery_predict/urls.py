from django.contrib import admin
from django.urls import path, include, re_path
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.views.static import serve
import os

urlpatterns = [
    # *** 1. ทางเดินสำหรับไฟล์ CSS/JS (ดักจับทุกทาง) ***
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),
    re_path(r'^surgery/static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),

    # 2. ทางเดินสำหรับ Admin
    path('admin/', admin.site.urls),
    
    # 3. Password Reset
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='registration/password_reset_form.html'), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='registration/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
    
    # 4. Main App
    path('', include('main.urls')), 
]
