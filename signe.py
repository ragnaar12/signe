import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
from utils import MyConvBlock
model = torch.load('model.pth', map_location=device)
model
next(model.parameters()).device
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')
show_image('data/asl_images/b.png')
image = tv_io.read_image('data/asl_images/b.png', tv_io.ImageReadMode.GRAY)
image
image.shape
IMG_WIDTH = 28
IMG_HEIGHT = 28

preprocess_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()  # From Color to Gray
])
processed_image = preprocess_trans(image)
processed_image
pshaperocessed_image.
plot_image = F.to_pil_image(processed_image)
plt.imshow(plot_image, cmap='gray')
batched_image = processed_image.unsqueeze(0)
batched_image.shape
batched_image_gpu = batched_image.to(device)
batched_image_gpu.device
output = model(batched_image_gpu)
output
prediction = output.argmax(dim=1).item()
prediction
# Alphabet does not contain j or z because they require movement
alphabet = "abcdefghiklmnopqrstuvwxy"
alphabet[prediction]

def predict_letter(file_path):
    show_image(file_path)
    image = tv_io.read_image(file_path, tv_io.ImageReadMode.GRAY)
    image = preprocess_trans(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    prediction = output.argmax(dim=1).item()
    # convert prediction to letter
    predicted_letter = alphabet[prediction]
    return predicted_letter

predict_letter("data/asl_images/b.png")

predict_letter("data/asl_images/a.png")

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
