from pathlib import Path

from src.svd_core import compress_image
from src.analysis import run_analysis
from src.visualization import (
    load_image,
    save_image,
    save_comparison
)


# Project paths
ROOT = Path(__file__).resolve().parent
image_name = input("Enter image filename: ").strip()
IMAGE_PATH = ROOT / "images" / image_name
RESULTS_PATH = ROOT / "results" / "results.csv"


def main():

    print("=" * 60)
    print("SVD IMAGE COMPRESSION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load input image
    # ---------------------------------------------------------

    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return

    original = load_image(IMAGE_PATH)

    print(f"\nImage loaded successfully.")
    print(f"Image dimensions: {original.shape[1]} x {original.shape[0]}")

    # ---------------------------------------------------------
    # 2. Select compression levels
    # ---------------------------------------------------------

    k_values = [10, 20, 50, 100]

    compressed_images = []
    labels = []

    # ---------------------------------------------------------
    # 3. Compress image using P1's SVD implementation
    # ---------------------------------------------------------

    print("\nGenerating compressed images...")

    for k in k_values:

        compressed = compress_image(original, k)

        compressed_images.append(compressed)
        labels.append(f"Rank {k}")

        output_name = f"compressed_{k}.png"

        output_path = save_image(
            compressed,
            output_name
        )

        print(f"  k = {k}: saved to {output_path}")

    # ---------------------------------------------------------
    # 4. Create visual comparison using P3's functions
    # ---------------------------------------------------------

    comparison_path = save_comparison(
        original,
        compressed_images,
        labels
    )

    print(f"\nComparison image saved to:")
    print(comparison_path)

    # ---------------------------------------------------------
    # 5. Run quantitative analysis using P2's implementation
    # ---------------------------------------------------------

    print("\nRunning compression and error analysis...")

    results = run_analysis(
        image_path=str(IMAGE_PATH),
        k_values=k_values,
        output_csv=str(RESULTS_PATH)
    )

    # ---------------------------------------------------------
    # 6. Final message
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 60)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("Compressed images saved in: outputs/")
    print("Analysis graphs saved in: outputs/")


if __name__ == "__main__":
    main()
