import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image # Used for basic image handling/error checking

# --- Configuration ---
IMAGE_FOLDER = 'images' # <-- Change this to your folder path
TOP_N_KEYWORDS = 3 # How many top keywords to display per image

# --- Load Pre-trained Model ---
print("Loading ResNet50 model...")
try:
    # Load the ResNet50 model with weights pre-trained on ImageNet
    model = ResNet50(weights='imagenet')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have a stable internet connection to download weights if needed.")
    exit()

# --- Process Images ---
print(f"\nGetting top {TOP_N_KEYWORDS} keywords for images in folder: {IMAGE_FOLDER}")

if not os.path.isdir(IMAGE_FOLDER):
    print(f"Error: Folder not found at {IMAGE_FOLDER}")
    exit()

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No supported image files found in the folder.")
    exit()

for image_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, image_file)

    try:
        # Load and resize the image
        img = image.load_img(img_path, target_size=(224, 224))

        # Convert the image to a numpy array
        x = image.img_to_array(img)

        # Expand dimensions to match model input shape (add batch dimension)
        x = np.expand_dims(x, axis=0)

        # Preprocess the image data (specific to ResNet50)
        x = preprocess_input(x)

        # Get predictions from the model
        predictions = model.predict(x, verbose=0) # verbose=0 suppresses prediction progress bar

        # Decode the predictions (get human-readable labels)
        # decode_predictions returns a list of lists, one inner list per image in the batch
        decoded_predictions = decode_predictions(predictions, top=TOP_N_KEYWORDS)[0]

        # Extract just the descriptions
        top_keywords = [description for _, description, _ in decoded_predictions]

        # Print the filename and the top keywords
        print(f"{image_file} {' '.join(top_keywords)}") # Using space as separator as requested

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        continue # Continue with the next image even if one fails

print("\nProcessing complete.")
