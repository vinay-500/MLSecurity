import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels 3, output 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size: 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size: 64 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size: 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def load_model(model_path):
    # Load the trained model from a file (you might want to save the global model after training)
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode for inference
    return model

def predict_image(model, image_path):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Example usage
model = load_model('/Users/vinay/Desktop/fed/global_model.pth')
predicted_class = predict_image(model, 'cat.jpg')
print(f"Predicted Class: {predicted_class}")
