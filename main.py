import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# pre-trained MobileNet
model = MobileNet(weights='imagenet', include_top=True)

def preprocess_frame(frame_path):
    img = image.load_img(frame_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array

def predict_and_check(frame_path, true_label='car'):
    img_array = preprocess_frame(frame_path)
    predictions = model.predict(img_array)
    top_prediction = tf.keras.applications.mobilenet.decode_predictions(predictions, top=1)[0][0]
    predicted_label = top_prediction[1].lower()
    is_correct = 'correct' if 'car' in predicted_label else 'incorrect'
    print(f"Frame {os.path.basename(frame_path)}: Predicted '{predicted_label}' - {is_correct}")
    return 1 if 'car' in predicted_label else 0

# Define frame directories
static_frames_dir = r'C:\Users\ynver\OneDrive\Desktop\Projects\static_frames' 
moving_frames_dir = r'C:\Users\ynver\OneDrive\Desktop\Projects\moving_frames'  

# File lists (50 each)
static_files = [f"frame_{i:03d}.jpg" for i in range(50)]
moving_files = [f"frame_{i:03d}.jpg" for i in range(50)]

# Process static frames
static_correct = 0
for file in static_files:
    frame_path = os.path.join(static_frames_dir, file)
    if not os.path.exists(frame_path):
        print(f"File not found: {frame_path}")
        continue
    static_correct += predict_and_check(frame_path)

# Process moving frames
moving_correct = 0
for file in moving_files:
    frame_path = os.path.join(moving_frames_dir, file)
    if not os.path.exists(frame_path):
        print(f"File not found: {frame_path}")
        continue
    moving_correct += predict_and_check(frame_path)

# accuracies
static_accuracy = (static_correct / 50) * 100
moving_accuracy = (moving_correct / 50) * 100
print(f"\nStatic Feed Accuracy (50 frames): {static_accuracy:.2f}% ({static_correct}/50 correct)")
print(f"Moving Feed Accuracy (50 frames): {moving_accuracy:.2f}% ({moving_correct}/50 correct)")
