import cv2
import numpy as np
import os
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from tensorflow.keras.utils import register_keras_serializable

# Custom Layer registration
@register_keras_serializable()
class CustomLayer(Conv2D):
    pass

# Define emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Check if JSON content is loaded correctly
if not loaded_model_json:
    print("Error: Empty JSON content")
    exit()

try:
    # Parse JSON string into a dictionary
    model_config = json.loads(loaded_model_json)

    # Create a Sequential model
    emotion_model = Sequential()

    # Add layers to the model based on the parsed JSON configuration
    for layer_config in model_config['config']['layers']:
        layer_class = globals()[layer_config['class_name']]
        if layer_config['class_name'] == 'Conv2D':
            # Remove unsupported arguments for Conv2D layer
            layer_config['config'].pop('batch_input_shape', None)
        layer = layer_class(**layer_config['config'])
        emotion_model.add(layer)

    print("Loaded emotion model successfully")
except Exception as e:
    print("Error loading model from JSON:", e)
    exit()

# Load weights into the model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model weights")

# Start the webcam feed
# cap = cv2.VideoCapture("Bhumika.mp4")
cap = cv2.VideoCapture(0)
while True:
    print("loop")
    # Read frame from video
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    
    # Check if frame reading was successful
    if not ret:
        break
    
    # Face detection
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Emotion prediction
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()