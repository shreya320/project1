🩺 Cough Sound COVID Detection using Deep Neural Decision Forest (DNDF)

This project detects whether a cough audio sample belongs to a COVID-positive or non-COVID individual using a trained deep learning model.
It uses Flask for the backend and an HTML interface for easy file uploads and predictions.

🚀 Features

Upload a cough sound (.wav) file directly from the browser

Real-time prediction using the trained DNDF model (dndf_best.h5)

Displays whether the cough is COVID or Non-COVID

Can be integrated with your own HTML frontend

🧠 Model

The backend loads a DNDF (Deep Neural Decision Forest) model trained on the CoughVid and Virufy datasets.
Model input: MFCC features extracted from the cough audio.
Output: Binary classification —

0 → Non-COVID

1 → COVID

📁 Project Structure
project/
│
├── app.py                # Flask backend
├── dndf_best.h5          # Trained model
├── templates/
│   └── tool.html         # Frontend HTML
├── static/               # Optional folder for CSS, JS, or assets
└── README.md             # Project description

⚙️ Setup Instructions
1. Install Dependencies

pip install flask tensorflow librosa numpy

2. Run the Flask Server
python app.py


This starts the app locally at
👉 http://127.0.0.1:5000/

3. Upload and Predict

Open the local link in your browser

Upload a .wav file (a cough sound)

Click Analyze

The result (COVID / Non-COVID) will appear on the page
