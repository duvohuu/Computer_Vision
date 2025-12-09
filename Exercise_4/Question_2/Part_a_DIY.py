import numpy as np

def PCA(X, num_components):
    # 1. Tính mean và center dữ liệu
    mean_vector = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_vector

    # 2. Tính covariance
    n = X.shape[1]
    covariance_matrix = (X_centered @ X_centered.T) / (n - 1)

    # 3. Lấy eigenvalues/eigenvectors
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)

    # 4. Sắp giảm dần
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    # 5. Lấy top-k eigenvectors
    U_k = eigvecs_sorted[:, :num_components]   
    
    # 6. Chiếu dữ liệu sang PCA space
    Z = U_k.T @ X_centered                      

    # 7. Khôi phục lại dữ liệu (reconstruction)
    X_reconstructed = (U_k @ Z) + mean_vector

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
    
    # CASE 1: 1 eigenvector
    reconstructed_1 = PCA(image, num_components=1)
    binary_1 = apply_threshold(reconstructed_1, threshold=150)
    
    print("\n=== PCA with 1 eigenvector (binary result) ===")
    print(binary_1)
    
    # CASE 2: 3 eigenvectors
    reconstructed_3 = PCA(image, num_components=3)
    binary_3 = apply_threshold(reconstructed_3, threshold=150)

    print("\n=== PCA with 3 eigenvectors (binary result) ===")
    print(binary_3)
     
if __name__ == "__main__":
    main()