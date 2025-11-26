import cv2
import numpy as np
import os
import json


script_dir = os.path.dirname(os.path.abspath(__file__))
calib_path = os.path.join(script_dir, "..", "Question_3", "calib_params.json")

LEFT_IMG_PATH  = os.path.join(script_dir, "images", "left.jpg")
RIGHT_IMG_PATH = os.path.join(script_dir, "images", "right.jpg")

pts_left = []
pts_right = []

B = 55  # Baseline in mm

# ======================
# TWO IMAGE CALIBRATION 
# ======================
print("=" * 60)
print("LOADING CAMERA CALIBRATION")
print("=" * 60)

if not os.path.exists(calib_path):
    raise FileNotFoundError(f"Không tìm thấy file calibration: {calib_path}")

with open(calib_path, 'r') as f:
    calib = json.load(f)
    
K_orig      = np.array(calib["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32)
K_new       = np.array(calib["new_camera_matrix"], dtype=np.float32)

imgL_raw = cv2.imread(LEFT_IMG_PATH)
imgR_raw = cv2.imread(RIGHT_IMG_PATH)
if imgL_raw is None or imgR_raw is None:
    raise SystemExit("Không đọc được ảnh LEFT/RIGHT, kiểm tra đường dẫn.")

h, w = imgL_raw.shape[:2]
assert imgR_raw.shape[:2] == (h, w), "Hai ảnh phải cùng kích thước!"

# Undistort về cùng ma trận nội K_new
imgL = cv2.undistort(imgL_raw, K_orig, dist_coeffs, None, K_new)
imgR = cv2.undistort(imgR_raw, K_orig, dist_coeffs, None, K_new)

# ======================
# CLICK EVENTS
# ======================
def click_left(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_left) < 3:
        pts_left.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        if len(pts_left) == 1:
            _image = cv2.putText(param, 'P1', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                 2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow("Left image", _image)
            print(f"✓ Điểm P1 TRÁI: ({x}, {y})")
        elif len(pts_left) == 2:
            cv2.line(param, pts_left[0], pts_left[1], (0, 255, 0), 2)
            _image = cv2.putText(param, 'P2', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow("Left image", _image)
            print(f"✓ Điểm P2 TRÁI: ({x}, {y})")
        elif len(pts_left) == 3:
            cv2.line(param, pts_left[1], pts_left[2], (255, 0, 0), 2)
            _image = cv2.putText(param, 'P3', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow("Left image", _image)
            print(f"✓ Điểm P3 TRÁI: ({x}, {y})")

def click_right(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_right) < 3:
        pts_right.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        if len(pts_right) == 1:
            _image = cv2.putText(param, 'P1', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                 2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow("Right image", _image)
            print(f"✓ Điểm P1 PHẢI: ({x}, {y})")
        elif len(pts_right) == 2:
            cv2.line(param, pts_right[0], pts_right[1], (0, 255, 0), 2)
            _image = cv2.putText(param, 'P2', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                 2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow("Right image", _image)
            print(f"✓ Điểm P2 PHẢI: ({x}, {y})")
        elif len(pts_right) == 3:
            cv2.line(param, pts_right[1], pts_right[2], (255, 0, 0), 2)
            _image = cv2.putText(param, 'P3', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                 2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.imshow("Right image", _image)
            print(f"✓ Điểm P3 PHẢI: ({x}, {y})")


def point3d(u_left, v_left, u_right, v_right, K, B):
    f_x = K[0,0]
    f_y = K[1,1]
    cx = K[0, 2]
    cy = K[1, 2]
    d = u_left - u_right
    
    if abs(d) < 1e-6:
        raise ValueError("Disparity ≈ 0 → không tính được độ sâu")

    Z = (f_x * B) / d 
    
    # Tọa độ X, Y (với principal point)
    X = (u_left - cx) * Z / f_x  
    Y = (v_left - cy) * Z / f_y  
    return np.array([X, Y, Z]), d


def rotation_matrix_to_euler_angles(R):
    """
    Chuyển đổi ma trận rotation R (3x3) sang góc Euler (Roll, Pitch, Yaw).
    Convention: ZYX (Yaw-Pitch-Roll)
    
    Returns:
        roll (rad): góc quay quanh trục X
        pitch (rad): góc quay quanh trục Y  
        yaw (rad): góc quay quanh trục Z
    """
    # Kiểm tra gimbal lock
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
        
    return roll, pitch, yaw


def compute_orientation_svd(P1, P2, P3):
    """
    Tính orientation (Roll, Pitch, Yaw) từ 3 điểm 3D sử dụng SVD.
    
    P1, P2, P3: np.array shape (3,) - tọa độ 3D (X, Y, Z)
    
    Returns:
        R: ma trận rotation 3x3
        roll, pitch, yaw: góc Euler (degrees)
    """
    # Vector từ P1 đến P2 (trục X local)
    v1 = P2 - P1
    v1 = v1 / np.linalg.norm(v1)  # normalize
    
    # Vector từ P1 đến P3
    v2 = P3 - P1
    
    # Trục Z local (vuông góc với mặt phẳng P1-P2-P3)
    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3)
    
    # Trục Y local (vuông góc với X và Z)
    v2 = np.cross(v3, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Ma trận rotation từ local frame sang world frame
    R_local = np.column_stack([v1, v2, v3])
    
    # Sử dụng SVD để tìm rotation matrix tối ưu (loại bỏ noise)
    U, S, Vt = np.linalg.svd(R_local)
    R = U @ Vt
    
    # Đảm bảo det(R) = 1 (proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # Chuyển sang góc Euler
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    
    return R, np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


if __name__ == "__main__":
    print("\nSTEREO 3D MEASUREMENT WITH ORIENTATION")
    print("=" * 60)
    print(f"Camera matrix K_new:\n{K_new}")
    print(f"Baseline: {B} mm)")
    print(f"Left image:  {LEFT_IMG_PATH}")
    print(f"Right image: {RIGHT_IMG_PATH}")
    print("=" * 60)
    
    left_img  = cv2.imread(LEFT_IMG_PATH)
    right_img = cv2.imread(RIGHT_IMG_PATH)
    
    if left_img is None or right_img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh!\n  {LEFT_IMG_PATH}\n  {RIGHT_IMG_PATH}")

    left_disp  = left_img.copy()
    right_disp = right_img.copy()

    cv2.namedWindow("Left image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right image", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("Left image",  600, int(600/1.5))
    cv2.resizeWindow("Right image", 600, int(600/1.5))

    cv2.setMouseCallback("Left image",  click_left,  left_disp)
    cv2.setMouseCallback("Right image", click_right, right_disp)

    print("\nHƯỚNG DẪN:")
    print("1. Click P1 (gốc) trên ảnh TRÁI và PHẢI")
    print("2. Click P2 (định hướng trục X) trên ảnh TRÁI và PHẢI")
    print("3. Click P3 (định hướng mặt phẳng) trên ảnh TRÁI và PHẢI")
    print("4. Nhấn phím bất kỳ để tính toán\n")

    cv2.imshow("Left image", left_disp)
    cv2.imshow("Right image", right_disp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts_left) < 3 or len(pts_right) < 3:
        print("❌ Chưa chọn đủ 3 điểm!")
        exit()

    # Tính toán 3D cho 3 điểm
    try:
        P1_3d, d1 = point3d(pts_left[0][0], pts_left[0][1], 
                            pts_right[0][0], pts_right[0][1], K_new, B)
        P2_3d, d2 = point3d(pts_left[1][0], pts_left[1][1], 
                            pts_right[1][0], pts_right[1][1], K_new, B)
        P3_3d, d3 = point3d(pts_left[2][0], pts_left[2][1], 
                            pts_right[2][0], pts_right[2][1], K_new, B)

        # Tính chiều dài P1-P2
        length_12 = np.linalg.norm(P2_3d - P1_3d)
        length_13 = np.linalg.norm(P3_3d - P1_3d)
        
        # Tính orientation bằng SVD
        R, roll, pitch, yaw = compute_orientation_svd(P1_3d, P2_3d, P3_3d)
        
        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐO 3D")
        print("=" * 60)
        print(f"P1 (gốc):  X={P1_3d[0]:.2f} mm, Y={P1_3d[1]:.2f} mm, Z={P1_3d[2]:.2f} mm")
        print(f"P2 (X):    X={P2_3d[0]:.2f} mm, Y={P2_3d[1]:.2f} mm, Z={P2_3d[2]:.2f} mm")
        print(f"P3 (mặt):  X={P3_3d[0]:.2f} mm, Y={P3_3d[1]:.2f} mm, Z={P3_3d[2]:.2f} mm")
        
        print(f"\nDisparity:")
        print(f"  P1: {d1:.2f} px")
        print(f"  P2: {d2:.2f} px")
        print(f"  P3: {d3:.2f} px")
        
        print(f"\nChiều dài:")
        print(f"  P1→P2: {length_12:.2f} mm = {length_12/10:.2f} cm")
        print(f"  P1→P3: {length_13:.2f} mm = {length_13/10:.2f} cm")
        
        print("\n" + "=" * 60)
        print("ORIENTATION (SVD-based)")
        print("=" * 60)
        print("Ma trận Rotation (R):")
        print(R)
        print(f"\nGóc Euler (ZYX convention):")
        print(f"  Roll  (quay quanh X): {roll:+7.2f}°")
        print(f"  Pitch (quay quanh Y): {pitch:+7.2f}°")
        print(f"  Yaw   (quay quanh Z): {yaw:+7.2f}°")
        
        # Kiểm tra orthogonality của R
        R_check = R @ R.T
        error = np.linalg.norm(R_check - np.eye(3))
        print(f"\nOrthogonality error: {error:.2e} (nên < 1e-6)")
        print(f"Determinant R: {np.linalg.det(R):.6f} (nên = 1)")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n❌ LỖI: {e}")
        print("Gợi ý: Kiểm tra xem có chọn đúng điểm tương ứng trên 2 ảnh không")