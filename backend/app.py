from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import io
import tensorflow as tf
   

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load parameter model
with open('soil_params_model.pkl', 'rb') as f:
    param_model = pickle.load(f)

# Load image model
image_model = tf.keras.models.load_model('soil_image_model.h5')

# Define class names corresponding to image model output classes
# Adjust these names according to your image dataset classes
IMAGE_MODEL_CLASS_NAMES = ['Red_soil', 'black_soil', 'clay_soil']

@app.route('/predict-params', methods=['POST'])
def predict_params():
    try:
        data = request.get_json()
        feature_names = ['pH', 'OrganicCarbon', 'Nitrogen', 'Phosphorus', 'Potassium', 'Sand', 'Silt', 'Clay']
        
        # Create a DataFrame with appropriate column names and input values
        input_df = pd.DataFrame([{feature: float(data.get(feature, 0)) for feature in feature_names}])
        
        prediction = param_model.predict(input_df)[0]
        return jsonify({'soil_type': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        img_file = request.files['image']
        if img_file.filename == '':
            return jsonify({'error': 'No selected image'}), 400

        # Read image file and preprocess
        img = Image.open(io.BytesIO(img_file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Use the size your model expects

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Predict with image model
        preds = image_model.predict(img_array)
        class_idx = np.argmax(preds[0])
        soil_type = IMAGE_MODEL_CLASS_NAMES[class_idx]

        return jsonify({'soil_type': soil_type})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

