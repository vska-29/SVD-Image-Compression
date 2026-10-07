import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import Person 1's mathematical core
try:
    from src.svd_core import compress_image
except ImportError:
    try:
        from svd_core import compress_image
    except ImportError:
        # Fallback implementation if Person 1's file is not yet integrated
        def compress_image(image: np.ndarray, k: int) -> np.ndarray:
            U, s, Vt = np.linalg.svd(image, full_matrices=False)
            k = max(1, min(k, len(s)))
            U_k = U[:, :k]
            s_k = s[:k]
            Vt_k = Vt[:k, :]
            return np.clip(np.dot(U_k * s_k, Vt_k), 0, 255)


# ============================================================================
# METRICS & FORMULAS (Person 2 Mathematical Core)
# ============================================================================

def calculate_compression_ratio(m: int, n: int, k: int):
    """
    Calculates original elements, compressed elements, and compression ratio.

    Theory:
      Original image matrix: m * n values
      Compressed rank-k representation requires storing:
        - U_k matrix: m * k values
        - Sigma_k diagonal: k values
        - V_k^T matrix: k * n values
      Total compressed storage = k * (m + n + 1)

      Compression Ratio (CR) = (m * n) / (k * (m + n + 1))
      Storage Saved (%) = (1 - (k * (m + n + 1)) / (m * n)) * 100
    """
    orig_elements = m * n
    comp_elements = k * (m + n + 1)
    
    if comp_elements == 0:
        return orig_elements, comp_elements, float('inf'), 0.0
    
    cr = orig_elements / comp_elements
    storage_saved_percent = max(0.0, (1.0 - (comp_elements / orig_elements)) * 100.0)
    return orig_elements, comp_elements, cr, storage_saved_percent


def calculate_reconstruction_errors(orig: np.ndarray, recon: np.ndarray):
    """
    Calculates quantitative reconstruction errors between original and reconstructed image.

    Metrics:
      1. Relative Frobenius Norm Error:
         ||A - A_k||_F / ||A||_F = sqrt(sum((A - A_k)^2)) / sqrt(sum(A^2))
      2. Mean Squared Error (MSE):
         (1 / (m * n)) * sum((A - A_k)^2)
      3. Root Mean Squared Error (RMSE):
         sqrt(MSE)
      4. Peak Signal-to-Noise Ratio (PSNR in dB):
         10 * log10( (MAX_I^2) / MSE )  where MAX_I = 255 for 8-bit pixels
    """
    diff = orig.astype(np.float64) - recon.astype(np.float64)
    
    # 1. Frobenius norm relative error
    frob_diff = np.linalg.norm(diff, 'fro')
    frob_orig = np.linalg.norm(orig.astype(np.float64), 'fro')
    rel_frobenius_error = frob_diff / (frob_orig + 1e-12)

    # 2. MSE & RMSE
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    # 3. PSNR
    if mse < 1e-12:
        psnr = 99.99  # Identical reconstruction
    else:
        max_pixel = 255.0
        psnr = 10.0 * np.log10((max_pixel ** 2) / mse)

    return float(rel_frobenius_error), float(mse), float(rmse), float(psnr)


# ============================================================================
# COMPREHENSIVE EXPERIMENT RUNNER
# ============================================================================

def run_analysis(image_path: str, k_values=None, output_csv="results/results.csv"):
    """
    Runs the full quantitative analysis pipeline over an input image for a list of k values.
    Saves results to CSV and generates comparison plots.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100, 150, 200]

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at '{image_path}'. Please check path.")

    pil_img = Image.open(image_path).convert('L')
    orig_matrix = np.array(pil_img, dtype=np.float64)
    m, n = orig_matrix.shape
    max_rank = min(m, n)

    print("=" * 75)
    print("SVD QUANTITATIVE COMPRESSION & ERROR ANALYSIS (Person 2)")
    print("=" * 75)
    print(f"Loaded image: {image_path} | Dimensions: {m} x {n} | Full Rank: {max_rank}")
    print(f"Original elements: {m * n:,} pixel values")
    print("-" * 75)

    valid_k = sorted(list(set([k for k in k_values if 1 <= k <= max_rank])))
    results = []
    
    header = f"{'k':>4} | {'Retained%':>9} | {'Comp Ratio':>10} | {'Saved%':>7} | {'Rel Err (Fro)':>13} | {'MSE':>8} | {'PSNR (dB)':>9}"
    print(header)
    print("-" * len(header))

    for k in valid_k:
        reconstructed = compress_image(orig_matrix, k)
        orig_elem, comp_elem, cr, saved_pct = calculate_compression_ratio(m, n, k)
        pct_retained = (k / max_rank) * 100.0
        rel_frob, mse, rmse, psnr = calculate_reconstruction_errors(orig_matrix, reconstructed)

        record = {
            "k": k,
            "original_elements": orig_elem,
            "compressed_elements": comp_elem,
            "compression_ratio": round(cr, 2),
            "storage_saved_percent": round(saved_pct, 2),
            "singular_values_retained_percent": round(pct_retained, 2),
            "relative_error_frobenius": round(rel_frob, 4),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "psnr_db": round(psnr, 2)
        }
        results.append(record)

        print(f"{k:>4} | {pct_retained:>8.1f}% | {cr:>9.2f}x | {saved_pct:>6.1f}% | {rel_frob:>13.4f} | {mse:>8.2f} | {psnr:>9.2f}")

    print("-" * 75)

    save_results_to_csv(results, output_csv)
    print(f"-> Successfully saved experimental data to: {output_csv}")
    plot_analysis_graphs(results, orig_matrix)

    return results


def save_results_to_csv(results, filepath):
    """Writes tabular analysis results into a clean CSV file."""
    if not results:
        return
    keys = list(results[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def plot_analysis_graphs(results, orig_matrix=None):
    """Generates and saves analytical graphs for report and presentation."""
    ks = [r["k"] for r in results]
    rel_errors = [r["relative_error_frobenius"] for r in results]
    psnrs = [r["psnr_db"] for r in results]
    crs = [r["compression_ratio"] for r in results]

    # Graph 1: Error vs k
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Number of Components (k)', fontsize=12)
    ax1.set_ylabel('Relative Frobenius Error', color=color, fontsize=12)
    ax1.plot(ks, rel_errors, 'o-', color=color, linewidth=2, label='Relative Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('PSNR (dB) - Image Quality', color=color, fontsize=12)
    ax2.plot(ks, psnrs, 's--', color=color, linewidth=2, label='PSNR (dB)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Reconstruction Error & Visual Quality vs. Number of Components (k)', fontsize=13, pad=15)
    fig.tight_layout()
    plt.savefig('outputs/error_vs_k.png', dpi=200)
    plt.close()
    print("-> Generated plot: outputs/error_vs_k.png")

    # Graph 2: Compression Ratio vs Components k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, crs, 'd-', color='tab:green', linewidth=2.2, label='Compression Ratio (x:1)')
    ax.axhline(1.0, color='gray', linestyle=':', label='Break-even line (1.0x)')
    ax.set_xlabel('Number of Components (k)', fontsize=12)
    ax.set_ylabel('Compression Ratio', color='tab:green', fontsize=12)
    ax.set_title('Storage Reduction (Compression Ratio) vs. Number of Components (k)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig('outputs/compression_vs_k.png', dpi=200)
    plt.close()
    print("-> Generated plot: outputs/compression_vs_k.png")


if __name__ == "__main__":
    sample_image = "images/input_image.jpg"
    if not os.path.exists(sample_image):
        os.makedirs("images", exist_ok=True)
        print("Note: 'images/input_image.jpg' not found. Creating a synthetic test pattern...")
        synthetic = np.zeros((200, 200), dtype=np.uint8)
        y, x = np.ogrid[:200, :200]
        synthetic = (np.sin(x / 10) * np.cos(y / 10) * 127 + 128).astype(np.uint8)
        Image.fromarray(synthetic).save(sample_image)

    run_analysis(
        image_path=sample_image,
        k_values=[5, 10, 20, 50, 100],
        output_csv="results/results.csv"
    )
