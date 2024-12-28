from flask import Flask, request, jsonify
import json
import torch
from torchvision import transforms
from model.model import PretrainedConvModel
from PIL import Image


app = Flask(__name__, static_url_path='/static')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class_names = ['dog','horse','elephant','butterfly','chicken','cat','cow','sheep','spider','squirrel']
model = PretrainedConvModel(
                num_hidden=512,
                num_classes=10
            )
state_dict = torch.load('./model/state_dict.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict['model'])
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Convert the file stream directly to an Image object
        image = Image.open(file.stream).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return jsonify({'prediction': class_names[predicted.item()]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Production
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    # Development
    #app.run(host='0.0.0.0', port=8080)
