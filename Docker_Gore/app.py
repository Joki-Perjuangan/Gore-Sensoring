from flask import Flask, request, jsonify, send_file
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained segmentation model
model = tf.keras.models.load_model('segment_model.h5')

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    return image

def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    return pred_mask[0]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image file
    img = Image.open(BytesIO(file.read()))
    original_size = img.size

    # Prepare the image for prediction
    target_size = (128, 128)  # Change target_size to match the model's expected input size
    prepared_image = prepare_image(img, target_size)

    # Predict the segmentation mask
    prediction = model.predict(prepared_image)
    mask = create_mask(prediction)

    # Resize mask to the original image size
    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)

    # Convert mask to boolean array
    mask = np.array(mask) > 0

    # Apply blur to the original image based on the mask
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=10))  # Adjust blur radius as needed
    img_np = np.array(img)
    img_blurred_np = np.array(img_blurred)
    
    # Combine the original and blurred images based on the mask
    img_combined_np = np.where(mask[..., None], img_blurred_np, img_np)
    img_combined = Image.fromarray(img_combined_np.astype(np.uint8))

    # Save combined image to a BytesIO object
    img_io = BytesIO()
    img_combined.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)