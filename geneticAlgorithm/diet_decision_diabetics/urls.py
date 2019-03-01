from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.getDiet),
    re_path(r'^calculate$', views.calculate_nutritional_needs)
]
