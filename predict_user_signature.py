# predict_user_signature.py

import os
import joblib
import numpy as np
from utils.helpers import preprocess_image, extract_hog_features, extract_cnn_features, fuse_features, get_cnn_model

# === Path to the saved model ===
MODEL_PATH = "model/signature_classifier.pkl"

def predict_signature(image_path):
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    # Load model and CNN
    model = joblib.load(MODEL_PATH)
    cnn_model = get_cnn_model()

    # Preprocess and extract features from the input image
    try:
        img = preprocess_image(image_path)
    except ValueError as e:
        print(e)
        return

    hog_feat = extract_hog_features(img)
    cnn_feat = extract_cnn_features(img, cnn_model)
    feature = fuse_features(cnn_feat, hog_feat)

    # Predict
    prediction = model.predict([feature])[0]
    label = "Genuine Signature" if prediction == 0 else "Forged Signature"
    print(f"🔎 Prediction: {label}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict signature authenticity using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the signature image")
    args = parser.parse_args()

    predict_signature(args.image_path)
