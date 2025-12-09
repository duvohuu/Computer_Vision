import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def find_similar_images_pca(template_path, folder_path, top_n=5, n_components=50):
    # BƯỚC 1: Đọc và chuẩn bị dữ liệu
    irow, icol = 128, 128
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    faces_list = []
    valid_files = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue   
        img_resized = cv2.resize(img, (icol, irow))
        img_flat = img_resized.flatten()
        faces_list.append(img_flat)
        valid_files.append(img_file)
    faces = np.array(faces_list, dtype=np.float64)
    
    # BƯỚC 2: Tính mean face
    m = np.mean(faces, axis=0)
    mean_face_img = m.reshape(irow, icol)
    
    # BƯỚC 3: Trừ mean - chuẩn hóa dữ liệu
    faces_mean = faces - m
    
    # BƯỚC 4: Tính eigenvectors và eigenvalues
    L = np.dot(faces_mean, faces_mean.T)
    eigenvalues, V = np.linalg.eig(L)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # BƯỚC 5: Tính principal components (eigenfaces)
    PC = np.dot(faces_mean.T, V)
    N = min(n_components, PC.shape[1])
    PC = PC[:, :N]
    
    # BƯỚC 6: Tạo signatures cho tất cả ảnh
    signatures = np.dot(faces_mean, PC)
    
    # BƯỚC 7: Xử lý template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return [], None, None, None, [], f"Không thể đọc template: {template_path}"
    
    recognize = cv2.resize(template, (icol, irow))
    rec = recognize.flatten() - m
    rec_weighted = np.dot(rec, PC)
    
    # BƯỚC 8: Tính Euclidean distance
    scores = np.zeros(len(valid_files))
    for i in range(len(valid_files)):
        scores[i] = np.linalg.norm(signatures[i, :] - rec_weighted)
    
    # BƯỚC 9: Tìm top N ảnh gần nhất
    idx = np.argsort(scores)
    
    top_images = []
    for i in range(min(top_n, len(idx))):
        img_name = valid_files[idx[i]]
        distance = scores[idx[i]]
        top_images.append((img_name, distance))
    
    # Tạo thông tin để print
    info = {
        'n_images': len(image_files),
        'n_loaded': faces.shape[0],
        'image_size': (irow, icol),
        'mean_shape': m.shape,
        'faces_mean_shape': faces_mean.shape,
        'L_shape': L.shape,
        'eigenvalues_shape': eigenvalues.shape,
        'top_5_eigenvalues': eigenvalues[:5],
        'PC_shape': (faces_mean.T.shape[0], N),
        'n_components': N,
        'signatures_shape': signatures.shape,
        'template_shape': recognize.shape,
        'template_sig_shape': rec_weighted.shape,
        'n_distances': len(scores),
        'n_top_results': len(top_images)
    }
    
    return top_images, mean_face_img, PC, signatures, valid_files, info

def main():
    # Cấu hình
    template_path = "template.png"
    folder_path = "BTPCA2025"
    top_n = 3
    n_components = 50
    
    # Thực hiện PCA và tìm ảnh
    results = find_similar_images_pca(template_path, folder_path, top_n=top_n, n_components=n_components)
    
    if len(results) == 6:
        top_images, mean_face, eigenfaces, signatures, valid_files, info = results
    else:
        print(results[5])
        return
    
    # In tất cả thông tin một lần
    print("\n" + "=" * 70)
    print(" " * 15 + "CHƯƠNG TRÌNH TÌM ẢNH TƯƠNG TỰ DÙNG PCA")
    print(" " * 20 + "(EIGENFACES APPROACH)")
    print("=" * 70)
    
    print("\nBƯỚC 1: ĐỌC VÀ CHUẨN BỊ DỮ LIỆU")
    print(f"Tìm thấy {info['n_images']} ảnh trong folder")
    print(f"Đã load {info['n_loaded']} ảnh")
   
    print("\nBƯỚC 2: TÍNH MEAN FACE")
    print(f"Mean face shape: {info['mean_shape']}")
    
    print("\nBƯỚC 3: TRỪ MEAN - CHUẨN HÓA DỮ LIỆU")
    print(f"Faces_mean shape: {info['faces_mean_shape']}")
    
    print("\nBƯỚC 4: TÍNH EIGENVECTORS VÀ EIGENVALUES")
    print(f"Covariance matrix L shape: {info['L_shape']}")
    print(f"Eigenvalues shape: {info['eigenvalues_shape']}")
    print(f"Top 5 eigenvalues: {info['top_5_eigenvalues']}")
    
    print("\nBƯỚC 5: TÍNH PRINCIPAL COMPONENTS (EIGENFACES)")
    print(f"Principal Components shape: {info['PC_shape']}")
    print(f"Giữ lại {info['n_components']} principal components")
    
    print("\nBƯỚC 6: TẠO SIGNATURES CHO TẤT CẢ ẢNH")
    print(f"Signatures shape: {info['signatures_shape']}")
    print(f"Mỗi ảnh được biểu diễn bởi {info['signatures_shape'][1]} coefficients")
    
    print("\nBƯỚC 7: XỬ LÝ TEMPLATE IMAGE")
    print(f"Template shape: {info['template_shape']}")
    print(f"Template signature shape: {info['template_sig_shape']}")
    
    print("\nBƯỚC 8: TÍNH EUCLIDEAN DISTANCE")
    print(f"Đã tính {info['n_distances']} distances")
    
    print("\nBƯỚC 9: TÌM TOP N ẢNH GẦN NHẤT")
    print(f"Đã tìm top {info['n_top_results']} ảnh giống nhất")
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ: TOP {} ẢNH GIỐNG TEMPLATE NHẤT".format(top_n))
    print("=" * 70)
    
    for i, (img_name, distance) in enumerate(top_images, 1):
        print(f"{i}. {img_name:<25} - Khoảng cách: {distance:.4f}")
    
    # Tạo hình ảnh kết quả
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        print(f"Không thể đọc template để hiển thị: {template_path}")
        return
    
    template_resized = cv2.resize(template_img, (128, 128))
    
    display_height = 300
    display_width = 300
    images = []
    temp_display = cv2.resize(template_resized, (display_width, display_height))
    temp_display_bgr = cv2.cvtColor(temp_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(temp_display_bgr, "TEMPLATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    images.append(temp_display_bgr)
    
    mean_display = cv2.resize(mean_face.astype(np.uint8), (display_width, display_height))
    mean_display_bgr = cv2.cvtColor(mean_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(mean_display_bgr, "MEAN FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    images.append(mean_display_bgr)
    for i, (img_name, distance) in enumerate(top_images, 1):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (display_width, display_height))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        text1 = f"Best match {i}"
        text2 = f"Dist: {distance:.2f}"
        cv2.putText(img_bgr, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_bgr, text2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        images.append(img_bgr)
    
    if len(images) >= 3:
        row1 = cv2.hconcat(images[:3])
    else:
        row1 = cv2.hconcat(images)
    
    if len(images) >= 6:
        row2 = cv2.hconcat(images[3:6])
        if row1.shape[1] > row2.shape[1]:
            padding = np.zeros((display_height, row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, padding])
        result = cv2.vconcat([row1, row2])
    elif len(images) >= 4:
        row2 = cv2.hconcat(images[3:])
        if row1.shape[1] > row2.shape[1]:
            padding = np.zeros((display_height, row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, padding])
        result = cv2.vconcat([row1, row2])
    else:
        result = row1
    
    cv2.imshow("PCA Face Recognition Results", result)
    print("\nNhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\nHoàn thành!")
if __name__ == "__main__":
    main()
