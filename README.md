# Garbage Waste Classification for Recycling

AI-powered waste classification system using Convolutional Neural Networks (CNNs) with heavy data augmentation to handle real-world conditions.

## Problem Statement

- Waste segregation is inefficient due to human error
- Public datasets contain clean, centered objects
- Real waste is often dirty, broken, or overlapping
- Lighting conditions vary widely
- Models confuse visually similar materials (plastic vs glass)

## Solution

Train CNNs with heavy augmentation and real-world cluttered images to improve robustness.

## Features

- **Custom CNN Architecture** - Deep network with batch normalization and dropout
- **Heavy Data Augmentation** - Handles dirty, broken, overlapping objects
- **Modern Web Interface** - Drag-and-drop image upload
- **6 Waste Categories** - Cardboard, Glass, Metal, Paper, Plastic, Trash
- **Recycling Tips** - Guidance for proper waste disposal
- **Training Diagnostics** - Per-epoch logging, precision/recall/F1 metrics
- **Best Model Checkpointing** - Automatic checkpoint on validation loss improvement

## Quick Start

### 1. Install Dependencies

```bash
cd "/Users/rishitmathur/Desktop/DL MINI"
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
cd app
python app.py
```

Then open **http://localhost:5001** in your browser.

### 3. (Optional) Train with Your Data

Organize your images in this structure:
```
data/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Then run:
```bash
cd model
python train.py --data-dir ../data --epochs 50
```

## Project Structure

```
DL MINI/
  app/
    app.py              # Flask server
    templates/
      index.html        # Web interface
    static/
      style.css         # Styling
  model/
    cnn_model.py        # CNN architecture
    train.py            # Training pipeline
    augmentation.py     # Data augmentation
  saved_models/         # Trained models
  checkpoints/          # Best model checkpoints
  data/                 # Training dataset
  requirements.txt      # Dependencies
  README.md
```

## Technologies

- **PyTorch** - Deep learning framework
- **Flask** - Web server
- **Albumentations** - Advanced image augmentation
- **PIL/OpenCV** - Image processing
- **scikit-learn** - Metrics (precision, recall, F1)

## Waste Categories

| Category  | Code | Recyclable |
|-----------|------|------------|
| Cardboard | CB   | Yes        |
| Glass     | GL   | Yes        |
| Metal     | MT   | Yes        |
| Paper     | PP   | Yes        |
| Plastic   | PL   | Yes        |
| Trash     | TR   | No         |

## API Endpoints

- `GET /` - Web interface
- `POST /classify` - Classify an image
- `GET /health` - Health check
- `GET /classes` - Get all categories

---

Built for a Sustainable Future
