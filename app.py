from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import os

app = Flask(__name__)

# -----------------------------
# Custom Layer Definition
# -----------------------------
class SoftDecisionTree(tf.keras.layers.Layer):
    def __init__(self, depth=4, **kwargs):
        super(SoftDecisionTree, self).__init__(**kwargs)
        self.depth = depth

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1], 7), 
                                initializer="random_normal",
                                trainable=True)
        self.w2 = self.add_weight(shape=(7,), 
                                initializer="zeros",
                                trainable=True)
        self.w3 = self.add_weight(shape=(8, 1), 
                                initializer="zeros",
                                trainable=True)
        super().build(input_shape)


    def call(self, inputs):
        return tf.matmul(inputs, self.w1)


# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = "dndf_v2.h5" 
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'SoftDecisionTree': SoftDecisionTree})
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# -----------------------------
# Feature Extraction Function 
# -----------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        print(f"Audio loaded: {file_path}")
        print(f"Audio length (samples): {len(y)}, Sample rate: {sr}")

        if len(y) < 22050: 
            print("Audio too short for analysis.")
            return None

        # Generate Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=120, n_fft=2048, hop_length=512
        )
        print(f"Mel-spectrogram shape before dB conversion: {mel_spec.shape}")

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"Mel-spectrogram shape after dB conversion: {mel_spec_db.shape}")

        # Pad or crop to (120, 122)
        if mel_spec_db.shape[1] < 122:
            pad_width = 122 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            print(f"Padded Mel-spectrogram to shape: {mel_spec_db.shape}")
        else:
            mel_spec_db = mel_spec_db[:, :122]
            print(f"Cropped Mel-spectrogram to shape: {mel_spec_db.shape}")

        # Add channel dimension
        features = mel_spec_db.reshape(1, 120, 122, 1)
        print(f"Final feature shape: {features.shape}")

        return features

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None



# -----------------------------
# Serve your frontend
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tool')
def tool():
    return render_template('tool.html')

# -----------------------------
# Prediction Route
# -----------------------------


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'Empty filename'}), 400

    file_path = 'temp_audio.wav'
    try:
        file.save(file_path)
        print(f"File saved to {file_path}")

        features = extract_features(file_path)
        if features is None:
            print("Feature extraction failed")
            return jsonify({'error': 'Error analyzing file'}), 500

        prediction = model.predict(features, verbose=0)
        print("Prediction raw output:", prediction)

        pred = prediction[0]  # shape (7,)
        print("Prediction vector:", pred)

        # Map 7-class prediction to Healthy / COVID
        if pred[0] >= max(pred[1:]):
            label_name = "Healthy"
            confidence = float(pred[0])
        else:
            label_name = "COVID"
            confidence = float(max(pred[1:]))

        result = f"Prediction: {label_name} ({confidence*100:.2f}%)"
        print("Result:", result)

    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print("Temporary file removed")

    return jsonify({'prediction': result})



if __name__ == '__main__':
    app.run(debug=True)
