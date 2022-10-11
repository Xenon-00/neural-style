import torch
from torchvision import transforms
import os
from architecture import NeuralStyle
from src.utils import load_image, plot_image, criterion


device = torch.device("cuda" if torch.cuda.is_available else 'cpu')

content_path = os.path.abspath(os.path.join('data', 'content', 'content.jpg'))
style_path = os.path.abspath(os.path.join('data', 'style', 'style.jpg'))

transform = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

content_image = load_image(content_path, transform).to(device)
style_image = load_image(style_path, transform).to(device)
output_image = load_image(content_path, transform).to(device)
output_image.requires_grad = True

model = NeuralStyle().to(device)
optimizer = torch.optim.AdamW([output_image], lr=0.05)

content_feature = model(content_image, layers=["4", "8", "13", "20"])
style_feature = model(style_image, layers=["4", "8", "13", "20"])

epochs = 3000
for epoch in range(1, epochs+1):
    output_feature = model(output_image, layers=["4", "8", "13", "20"])
    loss = criterion(content_feature, style_feature, output_feature, output_feature, style_weight=1e6)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    if epoch % 1000 == 0:
        print("Epoch : {} | Loss : {:.5f}".format(epoch, loss.item()))
        plot_image(output_image)