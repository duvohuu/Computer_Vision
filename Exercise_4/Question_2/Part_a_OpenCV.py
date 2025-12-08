import numpy as np
import cv2

def PCA_opencv(image, num_components):
    # Tính mean và eigenvectors sử dụng OpenCV PCACompute
    mean, eigenvectors = cv2.PCACompute(image, mean=None, maxComponents=num_components)

    # Giảm chiều dữ liệu (project xuống PCA space)
    Z = cv2.PCAProject(image, mean, eigenvectors)     

    # Phục hồi lại ảnh (reconstruction)
    X_reconstructed = cv2.PCABackProject(Z, mean, eigenvectors) 

    return X_reconstructed

def apply_threshold(matrix, threshold=150):
    return np.where(matrix >= threshold, 255, 0).astype(np.uint8)


def main():
    image = np.array([
        [0, 0, 0, 255, 0, 0, 0],
        [0, 255, 0, 255, 0, 0, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 0, 255, 255, 255, 255, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 255, 0, 255, 0, 0, 0],
        [0, 0, 0, 255, 0, 0, 0],
    ], dtype=np.float32)


    print("=== ORIGINAL MATRIX ===")
    print(image)

    # CASE 1 — PCA with 1 eigenvector
    reconstructed_1 = PCA_opencv(image, num_components=1)
    binary_1 = apply_threshold(reconstructed_1, threshold=150)

    print("\n=== PCA with 1 eigenvector (binary result) ===")
    print(binary_1)
    print(reconstructed_1)

    # CASE 2 — PCA with 3 eigenvectors
    reconstructed_3 = PCA_opencv(image, num_components=3)
    binary_3 = apply_threshold(reconstructed_3, threshold=150)

    print("\n=== PCA with 3 eigenvectors (binary result) ===")
    print(binary_3)
    print(reconstructed_3)


if __name__ == "__main__":
    main()
