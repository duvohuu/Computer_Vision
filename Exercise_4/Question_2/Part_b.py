import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ------------------------
# Utilities
# ------------------------
def compute_explained_variance_opencv(data, max_components=None):
    X = data.astype(np.float32)
    n_samples, n_features = X.shape
    if max_components is None:
        max_components = min(n_samples, n_features)

    mean, eigvecs = cv2.PCACompute(X, mean=None, maxComponents=max_components)
    Z = cv2.PCAProject(X, mean, eigvecs)

    variances = np.var(Z, axis=0, ddof=1)
    total_var = np.sum(np.var(X, axis=0, ddof=1))
    
    if total_var <= 0:
        explained = np.zeros_like(variances)
    else:
        explained = variances / total_var
    cumulative = np.cumsum(explained)
    return explained, cumulative, eigvecs, mean

def choose_k_by_threshold(cumulative, threshold=0.95):
    k = int(np.searchsorted(cumulative, threshold) + 1)
    if k < 1:
        k = 1
    if k > len(cumulative):
        k = len(cumulative)
    return k

def choose_k_elbow(cumulative):
    cum = np.asarray(cumulative, dtype=float)
    n = len(cum)
    if n == 0:
        return 1
    if n == 1:
        return 1
    x = np.arange(1, n+1)
    x1, y1 = 1, cum[0]
    x2, y2 = n, cum[-1]
    denom = np.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        return 1
    distances = np.abs((y2 - y1) * x - (x2 - x1) * cum + x2*y1 - y2*x1) / denom
    elbow_idx = int(np.argmax(distances))
    return elbow_idx + 1

def plot_and_save_cumulative(cumulative, out_path, title='Cumulative explained variance'):
    plt.figure(figsize=(7,4))
    k = len(cumulative)
    x = np.arange(1, k+1)
    plt.plot(x, cumulative, marker='o', label='Cumulative explained variance')
    plt.bar(x, np.diff(np.concatenate([[0.0], cumulative])), alpha=0.3, label='Per-component')
    plt.xlabel('Number of components (k)')
    plt.ylabel('Cumulative explained variance')
    plt.ylim(0,1.05)
    plt.grid(True)
    plt.axhline(0.90, color='red', linestyle='--', linewidth=0.7)
    plt.axhline(0.95, color='green', linestyle=':', linewidth=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ------------------------
# PCA pipeline using OpenCV + auto choose k
# ------------------------
def PCA_opencv_with_k_selection(image, threshold=0.95, max_components=None, result_dir='Question_2/Result'):
    # ensure result dir exists
    os.makedirs(result_dir, exist_ok=True)

    # Prepare data for OpenCV PCA: rows as samples (image is H x W already)
    data = image.astype(np.float32)

    # compute explained variance
    explained, cumulative, eigvecs, mean = compute_explained_variance_opencv(data, max_components=max_components)

    # save cumulative plot
    plot_and_save_cumulative(cumulative, os.path.join(result_dir, 'cumulative_explained_variance.png'))

    # choose k by threshold and elbow
    k_threshold = choose_k_by_threshold(cumulative, threshold=threshold)
    k_elbow = choose_k_elbow(cumulative)

    # For reconstruction we use k_threshold and k_elbow (we'll produce both)
    # Reconstruct function using OpenCV: need to re-run PCACompute with chosen k to get eigenvectors for that k
    def reconstruct_with_k(k):
        mean_k, eigvecs_k = cv2.PCACompute(data, mean=None, maxComponents=k)
        Z_k = cv2.PCAProject(data, mean_k, eigvecs_k)
        X_recon = cv2.PCABackProject(Z_k, mean_k, eigvecs_k)  
        X_recon_clipped = np.clip(X_recon, 0, 255).astype(np.uint8)
        return X_recon_clipped

    recon_threshold = reconstruct_with_k(k_threshold)
    recon_elbow = reconstruct_with_k(k_elbow)

    # save images
    cv2.imwrite(os.path.join(result_dir, f'reconstructed_k_threshold_{k_threshold}.png'), recon_threshold)
    cv2.imwrite(os.path.join(result_dir, f'reconstructed_k_elbow_{k_elbow}.png'), recon_elbow)

    return {
        'explained': explained,
        'cumulative': cumulative,
        'k_threshold': k_threshold,
        'k_elbow': k_elbow,
        'recon_threshold': recon_threshold,
        'recon_elbow': recon_elbow,
        'plot_path': os.path.join(result_dir, 'cumulative_explained_variance.png')
    }

# ------------------------
# Main
# ------------------------
def main():
    image_path = 'Question_2/image.png'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Couldn't load image: {image_path}")

    print("Loaded image shape:", img.shape)

    results = PCA_opencv_with_k_selection(img, threshold=1, max_components=None, result_dir='Question_2/Result')

    print(f"Suggested k by threshold (100%): {results['k_threshold']}")
    print(f"Suggested k by elbow heuristic : {results['k_elbow']}")
    print("Saved cumulative variance plot to:", results['plot_path'])
    print("Saved reconstructed and binary images in 'Question_2/Result'")

    # Visualize summary
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(results['recon_threshold'], cmap='gray'); axes[1].set_title(f'Recon (k={results["k_threshold"]})'); axes[1].axis('off')
    axes[2].imshow(results['recon_elbow'], cmap='gray'); axes[2].set_title(f'Recon (k={results["k_elbow"]})'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('Question_2/Result/PCA_opencv_auto_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
