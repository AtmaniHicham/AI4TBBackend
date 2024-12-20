from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://agreeable-wave-0c91d1a03.4.azurestaticapps.net"}})

model_weights = 'last.pt'
model = YOLO(model_weights)

@app.route('/')
def home():
    return "Server is running"

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if 'image' in data and 'confidence' in data:
        image_base64 = data['image']
        confidence = data['confidence']

        # Decode base64-encoded image
        try:
            image_data = base64.b64decode(image_base64)
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        except:
            return jsonify({'error': 'Invalid image format'}), 400

        # Run YOLO detection with specified confidence level
        results = model.predict(source=img, conf=confidence)

        # Extract Bacilli Counts and draw bounding boxes
        bacilli_count = 0
        for result in results:
            for detection in result.boxes.data:
                if int(detection[5]) == 0:  # Assuming bacillus class is 0 
                    bacilli_count += 1
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

        # Encode the image back to base64
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        response = {
            'bacilli_count': bacilli_count,
            'image': encoded_image
        }
        return jsonify(response)

    return jsonify({'error': 'No image or confidence level provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

