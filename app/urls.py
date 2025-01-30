from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    ]

#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