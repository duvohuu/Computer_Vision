import cv2
import numpy as np
import math
import json
import csv

# ================== CẤU HÌNH ==================
IMAGE_PATH  = r"D:\HK251\vision\BTL\Bai_2\pictures\img2.jpg"      # Ảnh cần đo card
CALIB_JSON  = r"D:\HK251\vision\BTL\Bai_3\calib_params.json"   # file JSON calib
POSE_INDEX  = 0      # dùng rvecs[POSE_INDEX], tvecs[POSE_INDEX]

CANNY_T1   = 50
CANNY_T2   = 150
KERNEL_SZ  = 3            # kernel cho opening/closing

# File CSV lưu 4 góc
OUTPUT_CSV = r"D:\HK251\vision\BTL\Bai_3\card_corners_uvXYZ.csv"


# ================== HÀM ĐỌC THAM SỐ CALIB ==================

def load_calib(json_path, pose_index=0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    rvecs = [np.array(r, dtype=np.float64) for r in data["rvecs"]]
    tvecs = [np.array(t, dtype=np.float64) for t in data["tvecs"]]

    if pose_index < 0 or pose_index >= len(rvecs):
        raise IndexError("POSE_INDEX nằm ngoài số lượng pose trong JSON")

    rvec = rvecs[pose_index]
    tvec = tvecs[pose_index].reshape(3, 1)   # (3,1)

    R, _ = cv2.Rodrigues(rvec)              # (3,3)

    return K, R, tvec

# ================== HÀM PIXEL → WORLD (MẶT PHẲNG Z_w = 0) ==================

def pixel_to_world_on_plane(u, v, K, R, T):
    """
    M_w = R^{-1}(K^{-1} z m_p - T)
    Giả sử mọi điểm nằm trên mặt phẳng Z_w = 0 (mặt phẳng bàn cờ),
    giải z (độ sâu trong hệ camera) sao cho thành phần Z_w = 0.
    """
    m_p = np.array([[u], [v], [1.0]], dtype=np.float64)   # (3,1)

    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)

    e3 = np.array([[0.0, 0.0, 1.0]])                      # (1,3)

    a = (e3 @ R_inv @ K_inv @ m_p)[0, 0]
    b = (e3 @ R_inv @ T)[0, 0]

    z_cam = b / a   # độ sâu trong hệ camera

    M_w = R_inv @ (K_inv @ (z_cam * m_p) - T)             # (3,1)

    return M_w.flatten()  # [X_w, Y_w, Z_w] (mm), Z_w ~ 0


# ================== MAIN ==================

def main():
    # ---- Load tham số calib từ JSON ----
    K, R, T = load_calib(CALIB_JSON, POSE_INDEX)
    print("Loaded calibration from JSON.")
    print("Camera matrix K:\n", K)
    print("Rotation matrix R:\n", R)
    print("Translation T:\n", T.ravel())

    # -------- BƯỚC 1: TIỀN XỬ LÝ + NHỊ PHÂN + CANNY --------
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("Không đọc được ảnh, kiểm tra lại IMAGE_PATH")

    img_draw = img.copy()   # giữ nguyên kích thước

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)

    # Card sáng trên nền tối
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Lọc nhiễu: opening + closing
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    thresh_open  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    thresh_clean = cv2.morphologyEx(thresh_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(thresh_clean, CANNY_T1, CANNY_T2)

    # -------- BƯỚC 2: TÌM CONTOUR LỚN NHẤT (CARD) --------
    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("Không tìm được contour.")
        return

    largest_cnt = max(contours, key=cv2.contourArea)
    #cv2.drawContours(img_draw, [largest_cnt], -1, (0, 0, 255), 2)

    # -------- BƯỚC 3: MIN AREA RECT + GÓC XOAY --------
    M = cv2.moments(largest_cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    rect = cv2.minAreaRect(largest_cnt)   # ((x,y), (w,h), angle)
    box  = cv2.boxPoints(rect)
    box  = np.int32(box)
    cv2.drawContours(img_draw, [box], 0, (0, 255, 0), 2)

    # Lấy cạnh dài nhất để tính góc ảnh
    max_len = -1
    dx_best, dy_best = 0, 0
    for i in range(4):
        x1, y1 = box[i]
        x2, y2 = box[(i + 1) % 4]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length > max_len:
            max_len = length
            dx_best, dy_best = dx, dy

    angle_rad = math.atan2(dy_best, dx_best)
    angle_deg = math.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    print("Tọa độ trọng tâm (cx, cy):", cx, cy)
    print("Góc xoay (có dấu):", angle_deg, "độ")

    cv2.circle(img_draw, (cx, cy), 5, (255, 0, 0), -1)
    text = f"Center: ({cx},{cy}) Angle: {angle_deg:.2f} deg"
    cv2.putText(img_draw, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -------- BƯỚC 4: TÍNH TỌA ĐỘ THỰC & CHIỀU DÀI / RỘNG --------

    # 4 góc pixel
    pixel_pts = [tuple(pt) for pt in box]  # [(u,v), ...]
    print("\nPixel corners (u, v) của hình:")
    for i, (u, v) in enumerate(pixel_pts):
        print(f"Corner {i}: u = {u}, v = {v}")

    # *** CHẤM 4 ĐIỂM GÓC LÊN ẢNH ***
    for i, (u, v) in enumerate(pixel_pts):
        cv2.circle(img_draw, (u, v), 7, (0, 255, 255), -1)  # chấm tròn
        cv2.putText(img_draw, str(i), (u + 5, v - 5),       # đánh số 0..3
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    # Chuyển 4 góc sang world (mm)
    world_pts = []
    for (u, v) in pixel_pts:
        Mw = pixel_to_world_on_plane(u, v, K, R, T)
        world_pts.append(Mw)
    world_pts = np.array(world_pts)   # (4,3)

    print("\nWorld coordinates (Xw, Yw, Zw) quy đổi từ 4 góc (mm):")
    for i, p in enumerate(world_pts):
        print(f"Corner {i}: Xw = {p[0]:.3f}, Yw = {p[1]:.3f}, Zw = {p[2]:.3f}")

    # ---- Lưu 4 góc ra CSV: u, v, Xw, Yw, Zw ----
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["corner_index", "u_pixel", "v_pixel",
                         "Xw_mm", "Yw_mm", "Zw_mm"])
        for i, ((u, v), (Xw, Yw, Zw)) in enumerate(zip(pixel_pts, world_pts)):
            writer.writerow([i, u, v,
                             f"{Xw:.6f}", f"{Yw:.6f}", f"{Zw:.6f}"])

    print("\nĐã lưu tọa độ 4 góc (pixel + world) vào CSV:")
    print(OUTPUT_CSV)

    # Độ dài 4 cạnh (mm)
    edge_lengths = []
    for i in range(4):
        p1 = world_pts[i]
        p2 = world_pts[(i + 1) % 4]
        d = np.linalg.norm(p2 - p1)
        edge_lengths.append(d)

    print("\nEdge lengths (mm) theo thứ tự box:")
    for i, d in enumerate(edge_lengths):
        print(f"Edge {i}-{(i+1)%4}: {d:.3f} mm")

    length_mm = max(edge_lengths)
    width_mm  = min(edge_lengths)

    print("\n>>> ƯỚC LƯỢNG KÍCH THƯỚC CARD (theo mặt phẳng bàn cờ):")
    print(f"Chiều dài (max cạnh): {length_mm:.3f} mm")
    print(f"Chiều rộng (min cạnh): {width_mm:.3f} mm")

    # -------- BƯỚC 5: HIỂN THỊ & LƯU ẢNH --------
    cv2.imshow("Thresh_clean", thresh_clean)
    cv2.imshow("Edges", edges)
    cv2.imshow("Result + measure", img_draw)

    cv2.imwrite("result_with_measure_and_corners.png", img_draw)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
