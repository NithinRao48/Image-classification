import numpy as np
import sys
import random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json

# Ensure the script receives the correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python test.py <image_directory> <output_directory>")
    sys.exit(1)

# Paths for image directory and model
IMDIR = sys.argv[1]
OUTDIR = sys.argv[2]
MODEL = os.path.join('models', 'resnet18.pth')

# Ensure the image directory exists
if not os.path.exists(IMDIR):
    print(f"Error: The directory '{IMDIR}' does not exist.")
    sys.exit(1)

# Ensure the output directory exists
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# Load the model for testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    model = torch.load(MODEL, map_location=device)
    model.eval()
except FileNotFoundError:
    print(f"Error: The model file '{MODEL}' does not exist.")
    sys.exit(1)

# Class labels for prediction
class_names = ['apple', 'atm card', 'cat', 'banana', 'bangle', 'battery', 'bottle', 'broom', 'bulb', 'calendar', 'camera']

# Retrieve images from directory
files = list(Path(IMDIR).resolve().glob('*.*'))

if len(files) == 0:
    print(f"Error: The directory '{IMDIR}' contains no images.")
    sys.exit(1)

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dictionary to store predictions
predictions = {}

# Perform prediction and save results
with torch.no_grad():
    for img_path in files:
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
        
        inputs = preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds.item()]
        
        # Save the prediction to the dictionary
        predictions[str(img_path)] = label
        
        # Create a plot for the image with the prediction
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Pred: " + label)
        ax.axis('off')
        
        # Save the plot to the output directory
        img_name = os.path.basename(img_path)
        save_path = os.path.join(OUTDIR, f"pred_{img_name}")
        fig.savefig(save_path)
        plt.close(fig)

# Save the predictions dictionary to a JSON file
json_path = os.path.join(OUTDIR, 'predictions.json')
with open(json_path, 'w') as json_file:
    json.dump(predictions, json_file, indent=4)

print(f"Predictions saved to {OUTDIR}")
print(f"JSON file saved to {json_path}")

'''
Sample run: python test.py test output
'''
