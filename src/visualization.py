from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Project folders
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(path):
    """Load an image and convert it to grayscale."""
    image = Image.open(path).convert("L")
    return np.array(image)


def display_original(image):
    """Display the original grayscale image."""
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.show()


def save_image(image, filename):
    """Save a reconstructed image."""
    image = np.clip(image, 0, 255).astype(np.uint8)
    output_path = OUTPUT_DIR / filename
    Image.fromarray(image).save(output_path)
    return output_path


def show_comparison(original, compressed_images, labels):
    """Display original and compressed images side by side."""

    images = [original] + compressed_images
    titles = ["Original"] + labels

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(4 * len(images), 4)
    )

    if len(images) == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def save_comparison(original, compressed_images, labels):
    """Save the comparison of original and compressed images."""

    images = [original] + compressed_images
    titles = ["Original"] + labels

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(4 * len(images), 4)
    )

    if len(images) == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "image_comparison.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# Test the visualization module
if __name__ == "__main__":

    input_path = IMAGE_DIR / "input.jpg"

    if not input_path.exists():
        print("ERROR: input.jpg was not found in the images folder.")

    else:
        original = load_image(input_path)

        print("Image loaded successfully.")
        print(f"Image dimensions: {original.shape[1]} x {original.shape[0]}")

        display_original(original)