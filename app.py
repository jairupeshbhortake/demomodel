from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load your trained plant disease detection model
model = load_model(r'C:\Users\JAI BHORTAKE\Desktop\data\model.keras')

# Define the directory to save uploaded images
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    
    # Save the uploaded image to the 'uploads' directory
    image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
    imagefile.save(image_path)

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Adjust size as per your model's input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Adjust shape for model input
    image = image / 255.0  # Normalize if your model was trained on normalized images

    # Make a prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class

    # Map the class index to the plant disease label
    class_labels = ["Healthy", "Powdery", "Rust"]  # Update with your actual labels
    classification = f'{class_labels[class_index]} ({prediction[0][class_index] * 100:.2f}%)'

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
