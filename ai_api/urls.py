from django.urls import path
from .views import SimplePrediction, HousePricePrediction

urlpatterns = [
    path('predict/', SimplePrediction.as_view(), name='simple-predict'),
    path('predict_house_price/', HousePricePrediction.as_view(), name='house-price-predict'),
]