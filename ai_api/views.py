from rest_framework.response import Response
from rest_framework.views import APIView
import tensorflow as tf
import numpy as np
import os

# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Create your views here.
class SimplePrediction(APIView):
    def get(self, request):
        # Simple TensorFlow operation: Predict y = 2x + 1 for x = 5
        model = lambda x: 2 * x + 1
        x = np.array([5])
        y_pred = model(x)
        return Response({"prediction": y_pred.tolist()})

class HousePricePrediction(APIView):
    def __init__(self):
        super().__init__()

        # Define the dataset
        self.X_train = np.array([600, 800, 1000, 1200, 1400], dtype=float)
        self.y_train = np.array([150, 200, 250, 300, 350], dtype=float)

        # Normalize data (scale down for better training)
        self.X_train = self.X_train / 1000  # Scale input to range ~0-1
        self.y_train = self.y_train / 1000  # Scale output to range ~0-1

        # Define the model properly
        inputs = tf.keras.Input(shape=(1,))
        outputs = tf.keras.layers.Dense(units=1)(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model with Adam optimizer
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss='mean_squared_error')

        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=1000, verbose=0)

    def get(self, request):
        house_size = request.GET.get("size", 1000)
        house_size = float(house_size) / 1000  # Normalize input

        house_size_array = np.array([[house_size]])  # Convert to NumPy array

        # Make prediction
        predicted_price = self.model.predict(house_size_array)[0][0] * 1000  # Convert back to original scale

        # Ensure valid number (no NaN)
        if np.isnan(predicted_price) or np.isinf(predicted_price):
            return Response({"error": "Prediction resulted in NaN or Inf."}, status=500)

        return Response({"house_size": house_size * 1000, "predicted_price": round(predicted_price, 2)})