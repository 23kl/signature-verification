import os
import numpy as np
from utils.helpers import (
    preprocess_image,
    extract_hog_features,
    extract_cnn_features,
    fuse_features,
    get_cnn_model
)
from model.train_model import train_hybrid_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === Dataset Paths ===
ORG_DIR = "data/full_org"
FORG_DIR = "data/full_forg"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTS)

def load_data():
    data = []
    labels = []
    cnn_model = get_cnn_model()  # Load CNN model once

    # Load original signatures
    for file in tqdm(os.listdir(ORG_DIR), desc="Loading Original"):
        if not is_image_file(file):
            continue
        img_path = os.path.join(ORG_DIR, file)
        try:
            img = preprocess_image(img_path)
            hog_feat = extract_hog_features(img)
            cnn_feat = extract_cnn_features(img, cnn_model)
            feature = fuse_features(cnn_feat, hog_feat)
            data.append(feature)
            labels.append(0)  # 0 = genuine
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    # Load forged signatures
    for file in tqdm(os.listdir(FORG_DIR), desc="Loading Forged"):
        if not is_image_file(file):
            continue
        img_path = os.path.join(FORG_DIR, file)
        try:
            img = preprocess_image(img_path)
            hog_feat = extract_hog_features(img)
            cnn_feat = extract_cnn_features(img, cnn_model)
            feature = fuse_features(cnn_feat, hog_feat)
            data.append(feature)
            labels.append(1)  # 1 = forgery
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    print("🔄 Loading data...")
    X, y = load_data()

    print("🧪 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(" Training model...")
    train_hybrid_model(X_train, y_train, X_test, y_test)
