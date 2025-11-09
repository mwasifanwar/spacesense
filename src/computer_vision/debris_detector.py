import torch
import torch.nn as nn
import torchvision.models as models

class DebrisDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DebrisDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.detection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4 + num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        outputs = self.detection_head(features)
        return outputs

class DetectionModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DebrisDetector().to(self.device)
        self.load_model()
    
    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.config.get('computer_vision.model_path'), map_location=self.device))
        except:
            pass
    
    def detect(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return self.process_predictions(predictions)
    
    def process_predictions(self, predictions):
        boxes = predictions[:, :4]
        scores = torch.softmax(predictions[:, 4:], dim=1)
        return boxes, scores
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)