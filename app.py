from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import time

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/sign_model.h5")

# Define sign labels
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
          "O", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Adjust if needed

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering
time.sleep(2)  # Allow camera to warm up

# Function to detect sign from video
def detect_sign(frame):
    img = cv2.resize(frame, (64, 64))  # Resize image
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize image
    predictions = model.predict(img, verbose=0)  # Disable verbose logs
    return labels[np.argmax(predictions)]

# Video capture function
def generate_frames():
    global cap

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detected_sign = detect_sign(frame)

        # Display detected sign
        cv2.putText(frame, detected_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
