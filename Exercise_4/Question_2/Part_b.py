import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    image = cv2.imread('Question_2/image.png', cv2.IMREAD_GRAYSCALE)

    print("=== ORIGINAL IMAGE ===")
    print(image)

    # CASE 1 — PCA with 1 eigenvector
    reconstructed_1 = PCA_opencv(image, num_components=1)
    binary_1 = apply_threshold(reconstructed_1, threshold=150)

    print("\n=== PCA with 1 eigenvector (binary result) ===")
    print(binary_1)

    # CASE 2 — PCA with 3 eigenvectors
    reconstructed_3 = PCA_opencv(image, num_components=55)
    binary_3 = apply_threshold(reconstructed_3, threshold=150)

    print("\n=== PCA with 3 eigenvectors (binary result) ===")
    print(binary_3)
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstructed_1, cmap='gray')
    axes[0, 1].set_title('Reconstructed (1 eigenvector)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(reconstructed_3, cmap='gray')
    axes[0, 2].set_title('Reconstructed (3 eigenvectors)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary_1, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Binary (1 eigenvector)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(binary_3, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title('Binary (3 eigenvectors)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('Question_2/Result/PCA_opencv_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to 'Question_2/Result/PCA_opencv_results.png'")
    plt.show()

    # Save individual images
    cv2.imwrite('Question_2/Result/reconstructed_1_eigen.png', reconstructed_1)
    cv2.imwrite('Question_2/Result/binary_1_eigen.png', binary_1)
    cv2.imwrite('Question_2/Result/reconstructed_3_eigen.png', reconstructed_3)
    cv2.imwrite('Question_2/Result/binary_3_eigen.png', binary_3)
    print("✓ Saved individual images")


if __name__ == "__main__":
    main()
