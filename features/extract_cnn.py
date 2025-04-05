import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_cnn_features(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.view(-1).numpy()
