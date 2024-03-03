import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import normalize

# Load the trained model
model = load_model('BrainTumorTypesImproved50EpochsCategorical.keras')

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = normalize(img_array, axis=1)  
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    print(result)
    predicted_class_index = np.argmax(result, axis=1)[0]
    print(predicted_class_index)

    class_labels = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']

    predicted_class = class_labels[predicted_class_index]
    print("Predicted class:", predicted_class)

image_path = 'C://Users//imsar//Desktop//BrainTumorDetection//datasets//Testing//pituitary_tumor//50.jpg'
predict_image(image_path)
