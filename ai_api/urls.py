from django.urls import path
from .views import SimplePrediction, HousePricePrediction

urlpatterns = [
    path('predict/', SimplePrediction.as_view(), name='simple-predict'),
    path('predict_house_price/', HousePricePrediction.as_view(), name='house-price-predict'), # http://127.0.0.1:8000/api/predict_house_price/?size=1600
]