import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, 90)

    def forward(self, images):
        # with torch.no_grad():
        features = self.clip_model.encode_image(images)
        embeddings = self.projection_head(features.float())
        return self.classifier(embeddings)
