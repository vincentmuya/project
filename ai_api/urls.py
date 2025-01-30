from django.urls import path
from .views import SimplePrediction

urlpatterns = [
    path('predict/', SimplePrediction.as_view(), name='simple-predict'),
]