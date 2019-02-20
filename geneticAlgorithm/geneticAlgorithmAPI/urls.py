''' URLS to Handle toogling of the image's color '''
from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^toto$', views.hola),
]
