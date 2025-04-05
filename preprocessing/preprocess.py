import cv2
import numpy as np
from skimage import exposure

def preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image / 255.0
