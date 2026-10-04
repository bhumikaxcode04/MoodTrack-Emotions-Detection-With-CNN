# MoodTrack – Emotion Detection With CNN

MoodTrack is an emotion detection project that uses **Convolutional Neural Networks (CNN)** and **computer vision** to recognize human facial expressions from images or a camera feed.

The project is trained using the **FER2013 facial expression dataset** and can classify emotions based on detected facial expressions.

## Features

* Detects human faces using computer vision
* Recognizes facial expressions using a CNN model
* Supports real-time emotion detection using a camera
* Uses the FER2013 dataset for training
* Saves the trained model for later prediction
* Displays the detected emotion during testing

## Emotions Detected

The model can recognize the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pillow
* Convolutional Neural Network (CNN)

## Project Structure

```text
MoodTrack/
│
├── data/
│   └── FER2013 Dataset
│
├── model/
│   ├── emotion_model.json
│   └── emotion_model.h5
│
├── TranEmotionDetector.py
├── TestEmotionDetector.py
└── README.md
```

## Required Packages

Install the required Python packages:

```bash
pip install numpy
pip install opencv-python
pip install keras
pip install tensorflow
pip install pillow
```

## Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset.

Download the dataset from Kaggle:

**FER2013 Dataset:**
https://www.kaggle.com/msambare/fer2013

After downloading, place the dataset inside the `data` folder of the project.

```text
MoodTrack/
└── data/
    └── FER2013/
```

## Training the Model

After placing the dataset in the correct location, run:

```bash
python TranEmotionDetector.py
```

Training time depends on your computer's processor and available resources.

After training, the model structure and weights will be generated:

```text
emotion_model.json
emotion_model.h5
```

Create a `model` folder and place both files inside it:

```text
model/
├── emotion_model.json
└── emotion_model.h5
```

> **Note:** The trained `.h5` model file can be large. If it exceeds GitHub's file-size limit, consider using Git LFS or provide instructions for generating the model locally.

## Run Emotion Detection

After training the model, run:

```bash
python TestEmotionDetector.py
```

The application will use the trained CNN model to detect facial expressions and display the predicted emotion.

## How It Works

```text
FER2013 Dataset
       ↓
Data Preprocessing
       ↓
CNN Model Training
       ↓
Trained Emotion Model
       ↓
Camera / Image Input
       ↓
Face Detection
       ↓
Emotion Prediction
       ↓
Displayed Emotion
```

## Example

The system detects a person's facial expression and predicts an emotion such as:

```text
Detected Emotion: Happy
```

## Project Purpose

MoodTrack was developed as a practical computer vision and deep learning project to understand:

* Image processing
* Face detection
* Convolutional Neural Networks
* Dataset preparation
* Model training
* Emotion classification
* Real-time computer vision

## Future Improvements

* Improve model accuracy
* Add a graphical user interface
* Store emotion history
* Generate emotion statistics
* Add support for multiple faces
* Deploy the model as a web application

## Author

**Bhumika Saraswat**

MoodTrack is developed as an academic and learning project focused on computer vision and emotion recognition.
