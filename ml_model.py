import cv2
import numpy as np
from keras.models import load_model

def load_and_predict(image_path):
    # Load the model
    model = load_model("/Users/khushjajoo/Desktop/ECE 545/archivefull")  # Replace with the actual path to your trained model

    # Read and preprocess the image
    new_image = cv2.imread(image_path)
    new_image = cv2.resize(new_image, (224, 224))
    new_image = new_image / 255.0

    # Perform inference
    predictions = model.predict(np.expand_dims(new_image, axis=0))

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    return predicted_class