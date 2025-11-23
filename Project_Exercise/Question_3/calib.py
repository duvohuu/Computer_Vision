import cv2
import numpy as np
import glob
import os
import json

# ================== CẤU HÌNH BÀN CỜ (CHECKERBOARD) ==================

# Kích thước bàn cờ (số GÓC NỘI bộ theo chiều ngang, dọc)
# Ví dụ: bàn cờ 8x11 ô vuông → 7x10 góc nội → CHECKERBOARD = (7, 10)
CHECKERBOARD = (7, 10)

# Điều kiện dừng cho hàm cornerSubPix
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Vector lưu các điểm 3D (trong không gian vật lý) cho mỗi ảnh bàn cờ
objpoints = []

# Vector lưu các điểm 2D (trong ảnh) tương ứng cho mỗi ảnh bàn cờ
imgpoints = []

# Chiều dài thực tế của cạnh 1 ô vuông trên bàn cờ (milimet)
square_size_mm = 25.0   # MỖI Ô = 25 x 25 mm

# Tọa độ 3D của các điểm góc trong hệ toạ độ bàn cờ (ĐƠN VỊ = mm)
# Dạng: (1, N, 3) với N = CHECKERBOARD[0] * CHECKERBOARD[1]
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# np.mgrid[0:cols, 0:rows] → (2, cols, rows)
# .T.reshape(-1, 2) → (N, 2) với N = cols * rows
# Nhân với square_size_mm để đưa sang đơn vị mm
objp[0, :, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
    .T.reshape(-1, 2) * square_size_mm
)

# ================== ĐƯỜNG DẪN ẢNH & FOLDER LƯU ==================

# Lấy đường dẫn thư mục chứa script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Folder chứa ảnh bàn cờ
image_dir = os.path.join(script_dir, "images")
images = glob.glob(os.path.join(image_dir, '*.jpg'))

if len(images) == 0:
    images = glob.glob(os.path.join(image_dir, '*.png'))

# Folder để lưu ảnh đã vẽ góc
save_dir = os.path.join(script_dir, "calib_images")
os.makedirs(save_dir, exist_ok=True)   # tạo nếu chưa có

# File JSON lưu kết quả calib
calib_json = os.path.join(script_dir, "camera_calibration.json")

print("Tìm được", len(images), "ảnh trong folder", image_dir)

# ================== HÀM HIỂN THỊ ẢNH FULLSCREEN (TÙY CHỌN) ==================

def previewImage(text, img):
    """Hiển thị ảnh full màn hình trong 2s (chỉ dùng để kiểm tra)."""
    cv2.namedWindow(text, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(text, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(text, img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# ================== VÒNG LẶP QUA TỪNG ẢNH BÀN CỜ ==================

if len(images) == 0:
    print("Không tìm thấy ảnh nào trong folder ảnh, kiểm tra lại đường dẫn.")
    raise SystemExit

img1 = None  # để tránh lỗi NameError nếu không tìm được bàn cờ

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Không đọc được ảnh: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm các góc bàn cờ
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        print(f"Detected corners in image {os.path.basename(fname)}")

        # Thêm tập điểm 3D (mm) và 2D (pixel)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        imgpoints.append(corners2)

        # Vẽ các điểm góc lên ảnh
        img_draw = cv2.drawChessboardCorners(
            img.copy(),
            CHECKERBOARD,
            corners2,
            ret
        )

        # (1) Nếu muốn xem từng ảnh để kiểm tra, mở comment:
        # previewImage("Pre-Calibration", img_draw)

        # (2) LƯU ẢNH ĐÃ VẼ GÓC VÀO FOLDER
        base = os.path.basename(fname)              # tên file gốc
        save_path = os.path.join(save_dir, f'corners_{base}')
        cv2.imwrite(save_path, img_draw)
        print("Saved:", save_path)

        # Giữ lại kích thước ảnh cuối cùng cho bước calibrate
        img1 = img.copy()

cv2.destroyAllWindows()

# Nếu không ảnh nào phát hiện được bàn cờ thì dừng
if img1 is None:
    print("Không ảnh nào phát hiện được bàn cờ, không thể calibrate.")
    raise SystemExit

# ================== CALIBRATE CAMERA ==================

h, w = img1.shape[:2]
print("Image Width, Height:", w, h)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    (w, h),
    None,
    None
)

newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx,
    dist,
    (w, h),
    1,
    (w, h)
)

# ================== IN KẾT QUẢ CHO DỄ ĐỌC ==================

# Thiết lập cách in của numpy cho gọn
np.set_printoptions(precision=4, suppress=True)

print("\n========== KẾT QUẢ CALIBRATION ==========\n")
print(f"Reprojection error (RMS) ~ {ret:.6f}\n")

# 1) Ma trận nội tại (Camera matrix)
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print("1) Camera matrix (K):")
print(mtx)
print(f"   -> fx = {fx:.4f}, fy = {fy:.4f}")
print(f"   -> cx = {cx:.4f}, cy = {cy:.4f}")
print()

# 2) Hệ số méo (distortion coefficients)
print("2) Distortion coefficients (k1, k2, p1, p2, k3, ...):")
print(dist.ravel())
if dist.size >= 5:
    k1, k2, p1, p2, k3 = dist.ravel()[:5]
    print(f"   -> k1 = {k1:.6f}, k2 = {k2:.6f}, p1 = {p1:.6f}, p2 = {p2:.6f}, k3 = {k3:.6f}")
print()

# 3) Ma trận camera mới (dùng để undistort)
print("3) New camera matrix (optimal K):")
print(newcam_mtx)
print()

# 4) Các pose (rvec, tvec) cho từng ảnh bàn cờ
print("4) Extrinsic parameters cho từng ảnh bàn cờ (ĐƠN VỊ = mm):")
for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs), start=1):
    R, _ = cv2.Rodrigues(rvec)  # đổi vector quay -> ma trận quay 3x3
    d = np.linalg.norm(tvec)    # khoảng cách từ camera đến góc (0,0) bàn cờ

    print(f"\n--- Ảnh #{i} ---")
    print("Rotation vector (rvec):", rvec.ravel())
    print("Rotation matrix (R):")
    print(R)
    print("Translation vector (tvec) [mm]:", tvec.ravel())
    print(f"Khoảng cách |T| ~ {d:.2f} mm")

print("\n==========================================\n")

# ================== LƯU CÁC MA TRẬN CẦN THIẾT RA JSON ==================

# Tính px_per_mm từ tất cả ảnh
all_dists = []
cols = CHECKERBOARD[0]
rows = CHECKERBOARD[1]

for corners in imgpoints:
    cs = corners.reshape(-1, 2)
    
    # Khoảng cách theo hàng
    for r in range(rows):
        for c in range(cols - 1):
            idx1 = r * cols + c
            idx2 = r * cols + c + 1
            p1 = cs[idx1]
            p2 = cs[idx2]
            all_dists.append(np.linalg.norm(p1 - p2))
    
    # Khoảng cách theo cột
    for r in range(rows - 1):
        for c in range(cols):
            idx1 = r * cols + c
            idx2 = (r + 1) * cols + c
            p1 = cs[idx1]
            p2 = cs[idx2]
            all_dists.append(np.linalg.norm(p1 - p2))

avg_px_per_square = float(np.mean(all_dists))
px_per_mm = avg_px_per_square / square_size_mm
mm_per_px = square_size_mm / avg_px_per_square

print("\n===== SCALE PIXEL ↔ MM =====")
print(f"Avg edge length: {avg_px_per_square:.4f} px (cho 1 cạnh ô {square_size_mm} mm)")
print(f"1 mm ≈ {px_per_mm:.4f} px")
print(f"1 px ≈ {mm_per_px:.6f} mm")
print("============================\n")

calib_data = {
    "image_width": int(w),
    "image_height": int(h),
    "checkerboard": list(CHECKERBOARD),
    "square_size_mm": float(square_size_mm),
    "rms_reprojection_error": float(ret),

    "camera_matrix": mtx.tolist(),          # K
    "dist_coeffs": dist.tolist(),           # hệ số méo
    "new_camera_matrix": newcam_mtx.tolist(),

    # Thêm px_per_mm để Question_6 dùng
    "px_per_mm": px_per_mm,
    "mm_per_px": mm_per_px,
    "avg_px_per_square": avg_px_per_square,

    # rvecs, tvecs cho từng ảnh
    "rvecs": [rvec.tolist() for rvec in rvecs],
    "tvecs": [tvec.tolist() for tvec in tvecs]
}

with open(calib_json, "w", encoding="utf-8") as f:
    json.dump(calib_data, f, indent=4)

print("Đã lưu tham số calib vào file JSON:", calib_json)