"""
Flask Web Application for Garbage Waste Classification

Features:
- Drag-and-drop image upload
- Real-time classification with confidence scores
- Transfer learning with pre-trained ResNet18
- Beautiful, modern UI
- REST API endpoint
"""

import os
import sys
import ssl

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import io
import base64

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from cnn_model import get_model, CLASSES
from augmentation import get_inference_transform

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model and device
model = None
device = None
transform = None

# Class info for recycling tips
CLASS_INFO = {
    'cardboard': {
        'icon': '📦',
        'color': '#8B4513',
        'recyclable': True,
        'tips': 'Flatten boxes and remove tape/labels. Keep dry.'
    },
    'glass': {
        'icon': '🍾',
        'color': '#4169E1',
        'recyclable': True,
        'tips': 'Rinse containers. Separate by color if required locally.'
    },
    'metal': {
        'icon': '🥫',
        'color': '#708090',
        'recyclable': True,
        'tips': 'Rinse cans. Aluminum and steel are both recyclable.'
    },
    'paper': {
        'icon': '📄',
        'color': '#F4A460',
        'recyclable': True,
        'tips': 'Keep clean and dry. No greasy or wet paper.'
    },
    'plastic': {
        'icon': '🧴',
        'color': '#20B2AA',
        'recyclable': True,
        'tips': 'Check recycling number. Rinse containers.'
    },
    'trash': {
        'icon': '🗑️',
        'color': '#2F4F4F',
        'recyclable': False,
        'tips': 'General waste. Cannot be recycled - goes to landfill.'
    }
}


def load_model():
    """Load the trained model with transfer learning"""
    global model, device, transform
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model with transfer learning (pre-trained ResNet18)
    # This gives meaningful predictions even without waste-specific training
    print("Loading pre-trained ResNet18 model with transfer learning...")
    model = get_model('transfer', pretrained=True)
    
    # Check for fine-tuned weights
    model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'garbage_classifier_finetuned.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded fine-tuned weights from: {model_path}")
        except Exception as e:
            print(f"Could not load fine-tuned weights: {e}")
            print("Using pre-trained ImageNet weights for transfer learning")
    else:
        print("Using pre-trained ImageNet weights (transfer learning)")
        print("Note: For best results, fine-tune on waste dataset using train.py")
    
    model = model.to(device)
    model.eval()
    
    # Setup transform
    transform = get_inference_transform()
    
    print("Model loaded successfully!")


def predict_image(image):
    """
    Predict the class of an image using transfer learning model
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Dictionary with prediction results
    """
    global model, device, transform
    
    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Apply transform
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    # Get all class probabilities
    all_probs = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    # Get class info
    class_name = CLASSES[predicted_class]
    info = CLASS_INFO[class_name]
    
    return {
        'class': class_name,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'icon': info['icon'],
        'color': info['color'],
        'recyclable': info['recyclable'],
        'tips': info['tips']
    }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    API endpoint for image classification
    
    Accepts:
        - File upload (multipart/form-data)
        - Base64 encoded image (JSON)
    
    Returns:
        JSON with classification results
    """
    try:
        image = None
        
        # Check for file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                image = Image.open(file).convert('RGB')
        
        # Check for base64 image
        elif request.is_json:
            data = request.get_json()
            if 'image' in data:
                # Remove data URL prefix if present
                image_data = data['image']
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if image is None:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get prediction
        result = predict_image(image)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/classes')
def get_classes():
    """Get all available classes and their info"""
    return jsonify({
        'classes': CLASSES,
        'class_info': CLASS_INFO
    })


# Initialize model on startup
with app.app_context():
    load_model()


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🗑️  Garbage Waste Classification System")
    print("="*50)
    print("\nStarting server...")
    print("Open http://localhost:5001 in your browser\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
