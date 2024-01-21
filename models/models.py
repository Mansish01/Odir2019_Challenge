from torchvision import models
from torchvision.models import densenet161
from torch import nn 


class Dense161Model(nn.Module):
    def __init__(self, num_labels):
        super(Dense161Model, self).__init__()

        # Load pre-trained DenseNet model
        self.densenet = models.densenet161(pretrained=True)

        # Freeze all parameters in the pre-trained model
        for paramater in self.densenet.parameters():
            paramater.requires_grad = False

        # Replace the final layer of the classifier
        in_features = self.densenet.classifier.in_features

        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_labels),
        )

    def forward(self, x):
        return self.densenet(x)

