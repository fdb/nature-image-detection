# 🍃 Nature image detection

This project uses the [CLIP model](https://openai.com/index/clip/) from [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) to classify images based on text descriptions.

## How it works

Instead of traditional image classification, it determines if an image depicts "Nature" – specifically focusing on natural waterways like rivers and canals – or "Not Nature" by comparing the image embedding to the embedding of predefined text prompts (e.g., "a photo of a river", "a photo of a building").

The image is then classified based on which set of prompts (Nature or Not Nature) receives the highest similarity score from the CLIP model. This allows for flexible classification based on conceptual definitions rather than fixed categories.

## Requirements

*   Python 3.8 or higher
*   The `uv` package manager
*   Images to classify in a folder (default `./images`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fdb/nature-image-detection
    cd nature-image-detection
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```

## Usage

1.  Place your images in the folder specified by `IMAGE_FOLDER` (default `./images`).
2.  Open your terminal in the project directory.
3.  Run the script:
    ```bash
    uv run main.py
    ```

The script will print the classification (Nature or Not Nature) and the top-matching text prompts for each image.
