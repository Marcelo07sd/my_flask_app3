from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/modelo_completo_100_.h5')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mapeo de etiquetas
labels = ['cartón', 'vidrio', 'metal', 'orgánico', 'papel', 'plástico', 'basura']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return send_from_directory('.', 'index7.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('file')
    predictions = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img_array = prepare_image(filepath)
            prediction = model.predict(img_array)
            predicted_label = labels[np.argmax(prediction[0])]
            predictions.append({'filename': file.filename, 'prediction': predicted_label})

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
