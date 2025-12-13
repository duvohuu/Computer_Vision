import numpy as np
import cv2

# =========================
# 1) Dữ liệu (6x4)
# =========================
H1 = np.array([
    [5,  5,  1,  3],
    [5, 10,  2,  4],
    [6,  8, 15,  5],
    [7, 10, 35, 20],
    [8, 30,  5, 10],
    [25,33, 44, 55]
], dtype=np.float32)

H2 = np.array([
    [5, 10,  8, 23],
    [6, 10,  9, 56],
    [7, 34, 15, 76],
    [8, 56, 35, 20],
    [9, 30, 23, 23],
    [25,33, 44, 55]
], dtype=np.float32)

H3 = np.array([
    [5, 34, 11, 78],
    [8, 10, 15, 98],
    [9, 34, 15, 23],
    [10,13, 35, 20],
    [11,30, 89, 78],
    [25,99, 11, 21]
], dtype=np.float32)

TEMPLATE = np.array([
    [5,  9,  7, 20],
    [5, 10,  7, 55],
    [7, 33, 15, 76],
    [8, 56, 35, 20],
    [8, 30, 22, 23],
    [25,31, 43, 54]
], dtype=np.float32)


# =========================
# 2) Hàm PCA + nhận dạng
# =========================
def as_row_sample(mat: np.ndarray) -> np.ndarray:
    """Chuyển ma trận (H x W) -> 1 sample dạng 1 x (H*W) cho PCA của OpenCV"""
    return mat.reshape(1, -1).astype(np.float32)

def pca_train_and_recognize(train_mats, test_mat, k=2):
    """
    train_mats: list các ma trận ảnh mẫu (cùng kích thước)
    test_mat  : ma trận ảnh kiểm tra (cùng kích thước)
    k         : số thành phần PCA giữ lại (maxComponents)
    """
    # Tạo data: mỗi ảnh là 1 hàng (num_samples x num_features)
    X = np.vstack([as_row_sample(m) for m in train_mats])  # (3 x 24)
    x_test = as_row_sample(test_mat)                       # (1 x 24)

    # PCA bằng OpenCV
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(X, mean=None, maxComponents=k)

    # Chiếu train và test lên không gian PCA
    proj_train = cv2.PCAProject(X, mean, eigenvectors)      # (3 x k)
    proj_test  = cv2.PCAProject(x_test, mean, eigenvectors) # (1 x k)

    # Khoảng cách Euclid trong không gian PCA
    dists = np.linalg.norm(proj_train - proj_test, axis=1)  # (3,)
    best_idx = int(np.argmin(dists))

    return {
        "mean": mean,
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "proj_train": proj_train,
        "proj_test": proj_test,
        "dists": dists,
        "best_idx": best_idx
    }


# =========================
# 3) Chạy cho k=2 và k=1
# =========================
labels = ["H1", "H2", "H3"]
train_set = [H1, H2, H3]

for k in (2, 1):
    out = pca_train_and_recognize(train_set, TEMPLATE, k=k)
    print(f"\n===== PCA recognition (k={k}) =====")
    print("Eigenvalues:", out["eigenvalues"].ravel())
    print("Proj(test):", out["proj_test"].ravel())
    print("Distances:")
    for i, d in enumerate(out["dists"]):
        print(f"  dist to {labels[i]} = {d:.6f}")
    print("=> Nearest:", labels[out["best_idx"]])
