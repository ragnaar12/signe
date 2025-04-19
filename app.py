from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Bienvenue sur HBA Signe !'

@app.route('/predict', methods=['POST'])
def predict():
    return {'prediction': 'a'}  # test simple



from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Bienvenue sur HBA Signe ! API de prédiction en langue des signes."



from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # traitement de l’image ici
    return "Résultat ici"


torch.save(model, "model.pth")

from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello depuis HBA Signe !"

@app.route('/predict', methods=['POST'])
def predict():
    return {'prediction': 'a'}  # temporairement



app = Flask(__name__)

# Charger ton modèle
model = torch.load('model.pth', map_location='cpu')

# Définir la transformation pour l'image
IMG_WIDTH = 28
IMG_HEIGHT = 28

preprocess_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()
])

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si l'image est dans la requête
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Charger et transformer l'image
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_trans(image).unsqueeze(0)
        
        # Prédiction avec le modèle
        output = model(image)
        prediction = output.argmax(dim=1).item()
        
        # Alphabet sans 'j' et 'z'
        alphabet = "abcdefghiklmnopqrstuvwxy"
        predicted_letter = alphabet[prediction]
        
        return jsonify({'prediction': predicted_letter})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
