# SVD Image Compression

## 1. Problem Statement

Digital images require large amounts of storage and bandwidth. This project aims to reduce the storage requirements of images while maintaining acceptable visual quality by using Singular Value Decomposition (SVD) to obtain a low-rank approximation of the image.

---

## 2. Objective

The objective of this project is to implement SVD-based image compression using Python and study the relationship between compression level and reconstructed image quality.

The project evaluates different numbers of retained singular values using:

- Compression ratio
- Storage reduction
- Reconstruction error
- MSE
- RMSE
- PSNR
- Visual comparison

---

## 3. Methodology

The complete image compression pipeline is:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Matrix Representation
     ↓
Singular Value Decomposition
     ↓
Select k Significant Components
     ↓
Low-Rank Reconstruction
     ↓
Compression & Error Analysis
     ↓
Visual Comparison
     ↓
Results
