"""Image preprocessing and digit prediction utilities."""

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image


model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Prepare an image for the trained digit recognition model."""
    with Image.open(img_path) as uploaded_image:
        rgb_image = uploaded_image.convert("RGB")
        resized_image = rgb_image.resize((224, 224))
        image_array = img_to_array(resized_image) / 255.0

    return image_array.reshape(1, 224, 224, 3)


def predict_result(image):
    """Return the digit with the highest prediction score."""
    prediction = model.predict(image)
    return np.argmax(prediction[0], axis=-1)
