import os
import csv
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# --- Configuration ---
IMAGE_FOLDER = 'images'
# You can choose a different CLIP model, e.g., "openai/clip-vit-large-patch14"
MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_N_CONTEXT_PROMPTS = 5 # How many top overall prompts to display for context

# Define text prompts for classification.
# Add more prompts here to cover various aspects of nature and non-nature.
NATURE_PROMPTS = [
    "a photo of nature",
    "a photo of a natural landscape",
    "a photo of a forest",
    "a photo of a river",
    "a photo of grass",
    "a photo of trees",
    "a photo of flowers",
    "a photo of a waterway"
    "a photo of clouds",
    "a photo of rocks",
    "a photo of soil",
    "a scene of the natural environment",
]

NON_NATURE_PROMPTS = [
    "a photo of a city",
    "a photo of a building",
    "a photo of a house",
    "a photo of a room",
    "a photo of furniture",
    "a photo of a car",
    "a photo of a road",
    "a photo of a person",
    "a photo of food",
    "a photo of an urban scene",
    "a photo of an industrial area",
    "a photo of food",
    "an abstract image",
    "a drawing",
    "a painting",
    "a screenshot",
    "text on a screen",
    "a close-up of an object",
    "a photo of technical equipment",
]

# Combine all prompts for the model to process
ALL_PROMPTS = NATURE_PROMPTS + NON_NATURE_PROMPTS

# --- Setup Device ---
# Use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load CLIP Model and Processor ---
print(f"Loading CLIP model: {MODEL_NAME}...")
try:
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection and model name.")
    exit()

# --- Process Images ---
print(f"\nClassifying images in folder: {IMAGE_FOLDER}")

if not os.path.isdir(IMAGE_FOLDER):
    print(f"Error: Folder not found at {IMAGE_FOLDER}")
    exit()

image_files = [f for f in sorted(os.listdir(IMAGE_FOLDER)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No supported image files found in the folder.")
    exit()

RESULTS_FILE = 'results.csv'
csv_writer = None
csv_file_handle = None # To close it if opened

try:
    csv_file_handle = open(RESULTS_FILE, 'w', newline='')
    csv_writer = csv.writer(csv_file_handle)
    # Write header
    csv_writer.writerow(['file_name', 'status', 'classification', 'best_prompt', 'probability'])
    print(f"Results will be written to {RESULTS_FILE}")
except IOError as e:
    print(f"Warning: Could not open '{RESULTS_FILE}' for writing: {e}. Results will not be saved to CSV.")
    # csv_writer remains None, csv_file_handle remains None

print(f"Comparing images against {len(ALL_PROMPTS)} text prompts.")

for image_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, image_file)

    try:
        # Load the image
        image_pil = Image.open(img_path).convert("RGB")

        # Preprocess image and tokenize text prompts
        # The processor handles resizing, normalization for the image
        # and tokenization/padding for the text
        inputs = processor(text=ALL_PROMPTS, images=image_pil, return_tensors="pt", padding=True)

        # Move inputs to the chosen device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # --- Get Predictions ---
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits (similarity scores) for image-text pairs
        # logits_per_image[i][j] is the similarity between image i and text j
        logits_per_image = outputs.logits_per_image

        # Convert logits to probabilities using softmax
        probs = logits_per_image.softmax(dim=1) # Softmax across the text prompts

        # --- Determine Classification ---
        # Get the index of the highest probability prompt
        best_prompt_idx = probs.argmax().item()
        best_prompt_text = ALL_PROMPTS[best_prompt_idx]
        best_prompt_prob = probs[0, best_prompt_idx].item()

        # Check if the best-scoring prompt is in our list of nature prompts
        is_nature = best_prompt_text in NATURE_PROMPTS

        # --- Get Top N Context Prompts ---
        # Get indices of top probabilities
        top_n_indices = torch.topk(probs, TOP_N_CONTEXT_PROMPTS, dim=1).indices[0].tolist()

        # Get the corresponding prompts and their probabilities
        top_n_info = [(ALL_PROMPTS[i], probs[0, i].item()) for i in top_n_indices]

        # --- Print Results ---
        classification = "nature" if is_nature else "not_nature"

        print(f"{image_file}: {classification}")
        print(f"  Highest Probability Match: '{best_prompt_text}' ({best_prompt_prob:.2f})")
        print(f"  Top {TOP_N_CONTEXT_PROMPTS} overall prompts:")
        for prompt, prob in top_n_info:
             print(f"    - '{prompt}': {prob:.2f}")

        if csv_writer:
            csv_writer.writerow([image_file, "ok", classification, best_prompt_text, f"{best_prompt_prob:.4f}"])

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        if csv_writer:
            csv_writer.writerow([image_file, "error", "", "", ""])
        continue # Continue with the next image even if one fails

if csv_file_handle: # If the file was attempted to be opened
    csv_file_handle.close()
    if csv_writer: # If writer was successfully created (file opened)
        print(f"\nImage classification results also saved to {RESULTS_FILE}")

print("\nProcessing complete.")
