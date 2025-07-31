#!/usr/bin/python3
#FireAlertAI webui backend

import os
import cv2
import jetson_inference
import jetson_utils
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

OLDMODEL = ""

UPLOAD_FOLDER = os.path.expanduser("~/FireAlertAI/web/uploads")
OUTPUT_FOLDER = os.path.expanduser("~/FireAlertAI/web/outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
MODEL_PATH = os.path.expanduser("~/FireAlertAI/models/FA2.onnx")
LABELS_PATH = os.path.expanduser("~/FireAlertAI/models/labels.txt")

net = jetson_inference.detectNet(network="ssd-mobilenet-v1", model=MODEL_PATH, labels=LABELS_PATH, input_blob="input_0", output_cvg="scores",output_bbox="boxes",threshold=0.3)
net.SetClusteringThreshold(0.4)

@app.route('/detect', methods=['POST'])
def detect_fire():
    try:
        global OLDMODEL
        global net
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        conf = float(request.form.get('confidence', 0.5))
        model_name = request.form.get('model_name', 'FA2.onnx').strip()

        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        input_dir = UPLOAD_FOLDER
        os.makedirs(input_dir, exist_ok=True)

        model_path = os.path.expanduser(f"~/FireAlertAI/models/{model_name}")
        labels_path = os.path.expanduser("~/FireAlertAI/models/labels.txt")

        print("-----\n" + OLDMODEL + "\n----")

        if(model_name != OLDMODEL):
            OLDMODEL = model_name
            net = jetson_inference.detectNet(network="ssd-mobilenet-v1", model=model_path, labels=labels_path, input_blob="input_0", output_cvg="scores", output_bbox="boxes", threshold=0.3)
            net.SetClusteringThreshold(0.4)

        filename = secure_filename(file.filename)
        input_path = os.path.join(input_dir, filename)
        file.save(input_path)

        img = jetson_utils.loadImage(input_path)
        detections = net.Detect(img)
        filtered_detections = [d for d in detections if d.Confidence >= conf]

        img_cv2 = cv2.imread(input_path)
        for detection in filtered_detections:
            left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
            label = net.GetClassDesc(detection.ClassID)
            confidence = detection.Confidence
            cv2.rectangle(img_cv2, (left, top), (right, bottom), (255, 0, 255), 4)
            cv2.putText(img_cv2, f"{label}: {confidence:.2f}",
                        (left + 20, top + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (198, 249, 1), 2)

        name, ext = os.path.splitext(filename)
        model_suffix = os.path.splitext(model_name)[0]
        output_filename = f"{name}_{model_suffix}{ext}"
        output_path = os.path.join(input_dir, output_filename)

        cv2.imwrite(output_path, img_cv2)
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error in detect_fire(): {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    models_dir = os.path.expanduser("~/FireAlertAI/models")
    models = [f for f in os.listdir(models_dir) if f.endswith(".onnx")]
    return jsonify(models)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
