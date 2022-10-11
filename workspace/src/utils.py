import torch
import os
from torch.nn.functional import mse_loss

from PIL import Image
import matplotlib.pyplot as plt

mean = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
std = torch.FloatTensor([[[0.229, 0.224, 0.225]]])

def load_image(filepath, transform):
    image = Image.open(filepath).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def _gram_matrix(image):
    n, c, h, w = image.shape
    image = image.view(n*c, h*w)
    gram = torch.mm(image, image.t())
    gram = gram.div(n*c*h*w)
    return gram

def criterion(content_feature, style_feature, output_content, output_style, content_weight=1, style_weight=1e6):
    content_loss = 0
    for c, o in zip(content_feature, output_content):
        content_loss += mse_loss(c, o)
    
    style_loss = 0
    for s, o in zip(style_feature, output_style):
        style_gram = _gram_matrix(s)
        output_gram = _gram_matrix(o)
        style_loss += mse_loss(style_gram, output_gram)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss

def plot_image(image_output):
    styled_image = image_output[0].permute(1, 2, 0).cpu().detach()
    styled_image = (styled_image * std) + mean
    styled_image.clamp_(0, 1)

    fig = plt.figure(figsize=(6,6))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(styled_image, aspect='auto')
    plt.show()




