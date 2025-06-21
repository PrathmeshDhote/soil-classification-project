from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import webbrowser # Import the webbrowser module
import threading # Import threading for a slight delay
import os # Import os module to handle paths

app = Flask(__name__)
CORS(app)

# Load parameter model
try:
    with open('soil_params_model.pkl', 'rb') as f:
        param_model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'soil_params_model.pkl' not found. Make sure it's in the same directory as app.py.")
    exit() # Exit if model is not found

# Load image model
try:
    image_model = tf.keras.models.load_model('soil_image_model.h5')
except Exception as e:
    print(f"Error loading 'soil_image_model.h5': {e}. Make sure it's in the same directory as app.py.")
    exit() # Exit if model is not found

# Define class names corresponding to image model output classes
IMAGE_MODEL_CLASS_NAMES = ['Alluvial Soil', 'Arid Soil', 'Black Soil', 'Forest Soil', 'Laterite Soil', 'Peaty Soil', 'Red Soil', 'Saline Soil']

# Crop recommendation dictionary - UPDATED STRUCTURE
crop_recommendations = {
    "Alluvial Soil": {
        "title": "🌾 1. Alluvial Soil", # Added emoji
        "found_in": "Indo-Gangetic plains (Punjab, Haryana, UP, Bihar, West Bengal)",
        "crops": {
            "Cereals": ["Wheat", "Rice", "Maize", "Barley"],
            "Cash Crops": ["Sugarcane", "Cotton", "Jute"],
            "Vegetables": ["Potato", "Onion", "Tomato"],
            "Fruits": ["Mango", "Banana", "Papaya"],
        }
    },
    "Black Soil": {
        "title": "🌿 2. Black Soil (Regur Soil)", # Added emoji
        "found_in": "Maharashtra, MP, Gujarat, Telangana, parts of Karnataka",
        "crops": {
            "Most Suitable": ["Cotton"],
            "Oilseeds": ["Soybean", "Sunflower", "Groundnut"],
            "Pulses": ["Tur", "Gram"],
            "Millets": ["Jowar", "Bajra"],
            "Cereals": ["Wheat"],
        }
    },
    "Red Soil": {
        "title": "🍁 3. Red Soil", # Added emoji
        "found_in": "Tamil Nadu, Karnataka, Odisha, Chhattisgarh, parts of AP",
        "crops": {
            "Pulses": ["Red gram", "Green gram"],
            "Oilseeds": ["Groundnut", "Castor", "Sesame"],
            "Millets": ["Ragi", "Bajra"],
            "Vegetables": ["Chillies", "Onion", "Brinjal"],
        }
    },
    "Laterite Soil": {
        "title": "🪨 4. Laterite Soil",
        "found_in": "Western Ghats, Kerala, Odisha, Assam",
        "crops": {
            "Plantation Crops": ["Tea", "Coffee", "Rubber"],
            "Spices": ["Black Pepper", "Cardamom"],
            "Fruits": ["Pineapple", "Banana", "Cashew nut"],
        }
    },
    "Arid Soil": { # Placeholder for structure consistency
        "title": "🏜️ Arid Soil",
        "found_in": "Arid and semi-arid regions of Rajasthan, Gujarat, and Haryana",
        "crops": {
            "Suitable Crops": ["Barley", "Jowar", "Bajra", "Pulses (with irrigation)"]
        }
    },
    "Saline Soil": { # Placeholder for structure consistency
        "title": "🧂 Saline Soil",
        "found_in": "Areas with poor drainage, coastal regions, and arid zones",
        "crops": {
            "Tolerant Crops": ["Barley", "Cotton", "Sugar beet", "Date palm", "Asparagus"]
        }
    },
    "Peaty Soil": { # Placeholder for structure consistency
        "title": "⚫ Peaty Soil",
        "found_in": "Humid regions, especially where there's accumulation of organic matter",
        "crops": {
            "Suitable Crops": ["Rice", "Jute", "Sugarcane (requires proper drainage)"]
        }
    },
    "Forest Soil": {
        "title": "🌳 8. Forest Soil",
        "found_in": "Himalayan regions, Western/Eastern Ghats",
        "crops": {
            "Fruits": ["Apples", "Pears", "Plums", "Peaches"],
            "Vegetables": ["Beans", "Peas", "Carrots"],
            "Plantation Crops": ["Spices", "Tea", "Coffee"],
            "Medicinal Plants": ["Amla", "Ashwagandha", "Shatavari"],
        }
    },
}

@app.route('/predict-params', methods=['POST'])
def predict_params():
    try:
        data = request.get_json()
        feature_names = ['pH', 'OrganicCarbon', 'Nitrogen', 'Phosphorus', 'Potassium', 'Sand', 'Silt', 'Clay']

        # Create a DataFrame with appropriate column names and input values
        input_df = pd.DataFrame([{feature: float(data.get(feature, 0)) for feature in feature_names}])

        # Get prediction directly (assumed to return class labels)
        prediction = param_model.predict(input_df)[0]  # Get the predicted soil type directly

        soil_type = prediction  # prediction is already a soil type string

        # Get the corresponding structured crop recommendation
        recommendation_data = crop_recommendations.get(soil_type, {
            "title": soil_type,
            "found_in": "Information not available.",
            "crops": {"Note": ["No specific crop recommendations available for this soil type."]}
        })

        return jsonify({
            'soil_type': soil_type,
            'crop_recommendation': recommendation_data # Send the structured data
        })
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

        # For image predictions, we will simply return the soil type
        return jsonify({'soil_type': soil_type})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def open_browser():
    # Define the absolute path to your HTML file
    # This path was provided by you: D:\soil-classification-project\frontend\soil-classification.html
    html_file_path = r"D:\soil-classification-project\frontend\soil-classification.html"

    # Convert the local path to a file URI for the browser
    # Replace backslashes with forward slashes and prepend 'file:///'
    file_uri = f"file:///{html_file_path.replace(os.sep, '/')}"

    # Give the Flask server a moment to start up before opening the browser
    # This runs the webbrowser.open_new in a separate thread after a delay
    threading.Timer(1.5, lambda: webbrowser.open_new(file_uri)).start()

if __name__ == '__main__':
    open_browser() # Call the function to open the browser
    app.run(debug=True, use_reloader=False)