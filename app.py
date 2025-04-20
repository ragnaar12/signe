from flask import Flask, request, jsonify
import torch
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms

app = Flask(__name__)

# Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth", map_location=device)
model.eval()

alphabet = "abcdefghiklmnopqrstuvwxy"
IMG_WIDTH = 28
IMG_HEIGHT = 28

preprocess = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400
    file = request.files["file"]
    image = tv_io.read_image(file.stream, tv_io.ImageReadMode.GRAY)
    image = preprocess(image).unsqueeze(0).to(device)
    output = model(image)
    pred = alphabet[output.argmax(dim=1).item()]
    return jsonify({"prediction": pred})


