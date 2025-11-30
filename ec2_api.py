import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

# Load your trained model
model = load_model('FV.h5')

# Mapping of class indices to labels
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum',
    6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant',
    12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon',
    18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear',
    24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans',
    30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

app = Flask(__name__)

def prepare_image(image_bytes):
    try:
        # Load image from bytes
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=-1)[0]
        return labels[class_idx].capitalize()
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return jsonify(error="No image file found in the request"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="Empty filename"), 400

    try:
        image_bytes = file.read()
        prediction = prepare_image(image_bytes)
        return jsonify(prediction=prediction)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
