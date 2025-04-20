from flask import Flask, request, render_template
import torch
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms

app = Flask(__name__)

# Charge le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)
model.eval()

alphabet = "abcdefghiklmnopqrstuvwxy"

# Prétraitement d'image
IMG_WIDTH = 28
IMG_HEIGHT = 28
preprocess_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()
])

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "Aucun fichier envoyé", 400
    file = request.files["file"]
    image = tv_io.read_image(file.stream, tv_io.ImageReadMode.GRAY)
    image = preprocess_trans(image).unsqueeze(0).to(device)
    output = model(image)
    prediction = output.argmax(dim=1).item()
    predicted_letter = alphabet[prediction]
    return f"Lettre prédite : {predicted_letter}"

if __name__ == "__main__":
    app.run(debug=True)

