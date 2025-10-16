
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

class LungCancerModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.class_names = ['normal', 'benign', 'malignant']
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image_bytes):
        """Predict from image bytes"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': self.class_names[prediction],
            'confidence': float(confidence),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(self.class_names, probabilities[0])
            }
        }

# Usage in your backend:
# model = LungCancerModel('lung_cancer_model.pth')
# result = model.predict(uploaded_file_bytes)
