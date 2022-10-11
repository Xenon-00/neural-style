from torchvision.models import vgg16
from torch import nn

class NeuralStyle(nn.Module):
    def __init__(self):
        super().__init__()

        model = vgg16(pretrained=True).eval()
        self.model = model.features
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, x, layers):
        features = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in layers:
                features.append(x)
        
        return features
