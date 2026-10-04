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


def svd_compress(image, rank):
    """Perform SVD compression using the specified rank."""

    U, S, Vt = np.linalg.svd(image, full_matrices=False)

    compressed = (
        U[:, :rank]
        @ np.diag(S[:rank])
        @ Vt[:rank, :]
    )

    return compressed


def calculate_compression_ratio(image_shape, rank):
    """Calculate theoretical SVD storage and compression ratio."""

    m, n = image_shape

    original_values = m * n
    compressed_values = (m * rank) + rank + (rank * n)

    compression_ratio = original_values / compressed_values

    return compression_ratio


def plot_compression_ratios(ranks, ratios):
    """Plot compression ratio for different SVD ranks."""

    fig = plt.figure(figsize=(7, 5))

    plt.plot(ranks, ratios, marker="o")

    plt.xlabel("SVD Rank")
    plt.ylabel("Compression Ratio")
    plt.title("SVD Rank vs Compression Ratio")

    plt.grid(True)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "compression_ratio.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    return output_path


if __name__ == "__main__":

    input_path = IMAGE_DIR / "input.jpg"

    if not input_path.exists():
        print("ERROR: input.jpg was not found.")

    else:

        original = load_image(input_path)

        print("Image loaded successfully.")
        print(
            f"Image dimensions: "
            f"{original.shape[1]} x {original.shape[0]}"
        )

        # Test different compression levels
        ranks = [50, 20, 10]

        compressed_images = []
        compression_ratios = []

        for rank in ranks:

            compressed = svd_compress(original, rank)

            compressed_images.append(compressed)

            ratio = calculate_compression_ratio(
                original.shape,
                rank
            )

            compression_ratios.append(ratio)

            print(
                f"Rank {rank}: "
                f"Compression ratio = {ratio:.2f}:1"
            )

            save_image(
                compressed,
                f"compressed_rank_{rank}.jpg"
            )

        # Display comparison
        labels = [
            "Rank 50",
            "Rank 20",
            "Rank 10"
        ]

        show_comparison(
            original,
            compressed_images,
            labels
        )

        # Save comparison
        save_comparison(
            original,
            compressed_images,
            labels
        )

        # Create compression ratio graph
        plot_compression_ratios(
            ranks,
            compression_ratios
        )

        print("Visualization completed.")