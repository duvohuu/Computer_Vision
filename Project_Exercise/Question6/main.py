import cv2
import numpy as np
import os
import json


script_dir = os.path.dirname(os.path.abspath(__file__))
calib_path = os.path.join(script_dir, "..", "Question_3", "camera_calibration.json")

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
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_left) < 2:
        pts_left.append((x, y))
        cv2.circle(param, (x, y), 3, (0, 255, 0), -1)
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

def click_right(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_right) < 2:
        pts_right.append((x, y))
        cv2.circle(param, (x, y), 3, (0, 255, 0), -1)
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


def point3d(u_left, v_left, u_right, v_right, K, B):
    f_x = K[0,0]
    f_y = K[1,1]
    cx = K[0, 2]
    cy = K[1, 2]
    d = u_left - u_right
    
    if abs(d) < 1e-6:
        raise ValueError("Disparity ≈ 0 → không tính được độ sâu")

    Z = (f_x * B) / d 
    
    # Tọa độ X, Y (normalize về center)
    X = (u_left) * Z / f_x  
    Y = (v_left) * Z / f_y  
    return np.array([X, Y, Z]), d

if __name__ == "__main__":
    print("\nSTEREO 3D MEASUREMENT")
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

    cv2.imshow("Left image", left_disp)
    cv2.imshow("Right image", right_disp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts_left) < 2 or len(pts_right) < 2:
        print("❌ Chưa chọn đủ 2 điểm!")
        exit()

    # Tính toán 3D
    try:
        P1_3d, d1 = point3d(pts_left[0][0], pts_left[0][1], pts_right[0][0], pts_right[0][1], K_new, B)
        P2_3d, d2 = point3d(pts_left[1][0], pts_left[1][1], pts_right[1][0], pts_right[1][1], K_new, B)

        length_m = np.linalg.norm(P1_3d - P2_3d)
    
        
        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐO 3D")
        print("=" * 60)
        print(f"P1 (3D): X={P1_3d[0]:.4f} mm, Y={P1_3d[1]:.4f} mm, Z={P1_3d[2]:.4f} mm")
        print(f"P2 (3D): X={P2_3d[0]:.4f} mm, Y={P2_3d[1]:.4f} mm, Z={P2_3d[2]:.4f} mm")
        print(f"\nDisparity P1: {d1:.2f} pixels")
        print(f"Disparity P2: {d2:.2f} pixels")
        print(f"\nChiều dài đoạn nối P1-P2: L = {length_m:.4f} mm")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n❌ LỖI: {e}")
        print("Gợi ý: Kiểm tra xem có chọn đúng điểm tương ứng trên 2 ảnh không")