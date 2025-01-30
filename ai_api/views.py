from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import tensorflow as tf
import numpy as np

# Create your views here.
class SimplePrediction(APIView):
    def get(self, request):
        # Simple TensorFlow operation: Predict y = 2x + 1 for x = 5
        model = lambda x: 2 * x + 1
        x = np.array([5])
        y_pred = model(x)
        return Response({"prediction": y_pred.tolist()})