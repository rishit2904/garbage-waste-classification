"""
Flask Web Application for Garbage Waste Classification

Features:
- Drag-and-drop image upload
- Real-time classification with confidence scores
- EfficientNet-B2 backbone (upgraded from ResNet18)
- 8-variant Test-Time Augmentation (TTA) at inference
- Clean, modern UI
- REST API endpoint

Task 5 upgrade: TTA averages softmax across 8 image variants
Task 1 upgrade: loads EfficientNet-B2 model type
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
from augmentation import get_inference_transform, get_test_time_augmentation

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model and device
model = None
device = None
transform = None

# Class info for recycling tips - no emojis, using text labels
CLASS_INFO = {
    'cardboard': {
        'icon': 'CB',
        'color': '#92400E',
        'recyclable': True,
        'tips': 'Flatten boxes and remove tape/labels. Keep dry.'
    },
    'glass': {
        'icon': 'GL',
        'color': '#1D4ED8',
        'recyclable': True,
        'tips': 'Rinse containers. Separate by color if required locally.'
    },
    'metal': {
        'icon': 'MT',
        'color': '#475569',
        'recyclable': True,
        'tips': 'Rinse cans. Aluminum and steel are both recyclable.'
    },
    'paper': {
        'icon': 'PP',
        'color': '#B45309',
        'recyclable': True,
        'tips': 'Keep clean and dry. No greasy or wet paper.'
    },
    'plastic': {
        'icon': 'PL',
        'color': '#0F766E',
        'recyclable': True,
        'tips': 'Check recycling number. Rinse containers.'
    },
    'trash': {
        'icon': 'TR',
        'color': '#374151',
        'recyclable': False,
        'tips': 'General waste. Cannot be recycled - goes to landfill.'
    }
}


def load_model():
    global model, device, transform

    import os
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try multiple possible checkpoint locations
    CHECKPOINT_CANDIDATES = [
        "best_model.pth",
        "models/best_model.pth",
        "checkpoints/best_model.pth",
        "resnet_best.pth",
        "../saved_models/garbage_classifier.pth"
    ]

    model = get_model("efficientnet")  # keep whatever architecture name your code uses
    model = model.to(device)
    model.eval()

    loaded = False
    for ckpt_path in CHECKPOINT_CANDIDATES:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[OK] Loaded checkpoint from {ckpt_path}, epoch={checkpoint.get('epoch','?')}, acc={checkpoint.get('best_acc','?')}")
            else:
                model.load_state_dict(checkpoint)
                print(f"[OK] Loaded raw state dict from {ckpt_path}")
            loaded = True
            break

    if not loaded:
        print("[FATAL] No checkpoint found. App is running on random weights.")
        print(f"[FATAL] CWD = {os.getcwd()}")
        print(f"[FATAL] Files = {os.listdir('.')}")

    transform = get_test_time_augmentation()


def predict_image(image):
    """
    Predict the class of an image using EfficientNet-B2 with 8-variant TTA.

    Task 5: Runs inference over 8 augmented views of the image
    (original, hflip, vflip, 90°/180°/270° rot, hflip+90°, brightness+0.1)
    and averages the softmax outputs before taking argmax.

    Args:
        image: PIL Image or numpy array

    Returns:
        Dictionary with prediction results (same keys as before)
    """
    global model, device, transform

    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    # Task 5: 8-variant TTA — average softmax over all variants
    with torch.no_grad():
        all_probs_stack = []
        for tta_transform in transform:
            transformed = tta_transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]  # shape [num_classes]
            all_probs_stack.append(probs)

        # Average across all 8 augmentation variants
        avg_probs = torch.stack(all_probs_stack, dim=0).mean(dim=0)
        predicted_class = torch.argmax(avg_probs).item()
        confidence = avg_probs[predicted_class].item()

    # Get all class probabilities from averaged output
    all_probs = {CLASSES[i]: float(avg_probs[i]) for i in range(len(CLASSES))}

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
    print("  Garbage Waste Classification System")
    print("="*50)
    print("\nStarting server...")
    print("Open http://localhost:5001 in your browser\n")

    app.run(host='0.0.0.0', port=5001, debug=True)
