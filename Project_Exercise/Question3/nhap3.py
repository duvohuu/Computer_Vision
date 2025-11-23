import cv2
import numpy as np
import json
import os

# ================== CẤU HÌNH ==================
CALIB_JSON = r"D:\HK251\vision\BTL\Bai_3\calib_params.json"

# Ảnh CHESSBOARD chụp ở vị trí CAMERA MỚI
# (ví dụ: camera TRÁI, cùng vị trí sau này bạn sẽ chụp card)
POSE_IMAGE = r"D:\HK251\vision\BTL\Bai_3\left_pose_chessboard.jpg"

# Kích thước bàn cờ bạn đang dùng (nhớ đúng với calib gốc)
# CHECKERBOARD = (số_cột_góc_nội, số_hàng_góc_nội)
CHECKERBOARD = (7, 10)

# Cạnh 1 ô vuông (mm) – phải giống thông số calib gốc
SQUARE_SIZE_MM = 25.0

# cornerSubPix criteria
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ================== HÀM SINH ĐIỂM 3D BÀN CỜ ==================
def generate_objpoints(pattern_size, square_size_mm):
    """
    Sinh toạ độ world (X,Y,Z) cho chessboard Z = 0
    pattern_size: (cols, rows) = (số_cột_góc_nội, số_hàng_góc_nội)
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp  # (N, 3)


# ================== MAIN ==================
def main():
    # ---- Đọc file calib JSON ----
    if not os.path.isfile(CALIB_JSON):
        raise FileNotFoundError(f"Không tìm thấy file calib: {CALIB_JSON}")

    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    print("Đọc calib xong.")
    print("K =\n", K)
    print("dist_coeffs =", dist.ravel())

    # ---- Đọc ảnh chessboard ở pose mới ----
    img = cv2.imread(POSE_IMAGE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh chessboard pose mới: {POSE_IMAGE}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Tìm góc bàn cờ ----
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_FAST_CHECK
    )

    if not ret:
        print("Không detect được chessboard trong ảnh pose mới.")
        return

    # refine sub-pixel
    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=CRITERIA
    )

    # vẽ kiểm tra
    vis = img.copy()
    cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_sub, ret)
    cv2.imshow("Chessboard pose mới", vis)
    cv2.waitKey(500)   # nhìn thoáng qua 0.5s

    # ---- Chuẩn bị objpoints 3D ----
    objp = generate_objpoints(CHECKERBOARD, SQUARE_SIZE_MM)

    # ---- solvePnP: tính rvec, tvec cho POSE MỚI ----
    ok, rvec, tvec = cv2.solvePnP(
        objp,           # (N,3)
        corners_sub,    # (N,1,2)
        K,
        dist
    )
    if not ok:
        print("solvePnP thất bại.")
        return

    print("\nPose mới (rvec, tvec):")
    print("rvec =", rvec.ravel())
    print("tvec =", tvec.ravel())

    # ---- Append vào JSON ----
    if "rvecs" not in data or "tvecs" not in data:
        raise KeyError("JSON không có trường 'rvecs' hoặc 'tvecs' như calib gốc.")

    data["rvecs"].append(rvec.tolist())
    data["tvecs"].append(tvec.reshape(3, 1).tolist())

    new_pose_index = len(data["rvecs"]) - 1

    # Lưu lại JSON
    with open(CALIB_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("\nĐÃ THÊM POSE MỚI VÀO calib_params.json")
    print(f"POSE_INDEX mới dành cho vị trí CAMERA hiện tại là: {new_pose_index}")
    print("Hãy dùng POSE_INDEX này trong code đo card cho các ảnh chụp ở pose này.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
