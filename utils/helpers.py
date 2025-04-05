import cv2
import numpy as np
from skimage.feature import hog
import torch
from torchvision import models, transforms

# Define image preprocessing transform for CNN
cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # ImageNet mean/std
])

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_hog_features(img, size=(128, 128)):
    resized = cv2.resize(img, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    features = hog(gray, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)
    return features

def get_cnn_model():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
    model.eval()
    return model

def extract_cnn_features(img, model):
    img_tensor = cnn_transform(img).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        features = model(img_tensor)
    return features.view(-1).numpy()  # Flatten

def fuse_features(cnn_feat, hog_feat):
    return np.concatenate((cnn_feat, hog_feat))
