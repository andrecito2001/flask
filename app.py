from flask import Flask, request, jsonify
import cv2
import numpy as np
import os 

app = Flask(__name__)
 
# Carga el modelo de OpenCV
prototxt_path = 'MobileNetSSD_deploy.prototxt.txt'
model_path = 'MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
 
# Definición de clases
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
 
@app.route('/predict', methods=['POST'])
def predict():
    # Recibe la imagen
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Preprocesamiento de la imagen
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # Pasar el blob a través de la red y obtener las detecciones y predicciones
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filtra las detecciones por confianza
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            results.append({"label": CLASSES[idx], "confidence": float(confidence), "box": [startX, startY, endX, endY]})
    return jsonify(results)
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.', port=port)