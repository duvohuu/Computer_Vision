import cv2 as cv
import numpy as np
import json

# ================== CẤU HÌNH ==================
CALIB_JSON = r"D:\HK251\vision\BTL\Bai_3\calib_params.json"
LEFT_IMG   = r"D:\HK251\vision\BTL\Bai_6\images\moi\left3.jpg"
RIGHT_IMG  = r"D:\HK251\vision\BTL\Bai_6\images\moi\right3.jpg"

BASELINE_MM   = 60.0      # khoảng dịch cam theo trục X (mm)
DISPLAY_SCALE = 0.6       # scale hiển thị
WIN        = 31           # tham số KLT
MAX_LEVEL  = 4
FB_THRESH  = 2.0          # ngưỡng forward–backward error (px)

# ================== HÀM TÍNH 3D ==================
def pixel_to_3d(uL, vL, uR, vR, K, baseline_mm):
    """
    Từ pixel trái/phải (uL,vL,uR,vR) → toạ độ 3D (X,Y,Z) trong hệ camera trái.
    baseline_mm: khoảng cách hai tâm camera, đơn vị mm.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    d = float(uL - uR)          # disparity ngang (px)
    if d == 0:
        raise ValueError("Disparity d = 0 → không tính được Z.")

    Z = fx * baseline_mm / abs(d)  # mm (dùng |d| để tránh âm)

    X = (uL - cx) * Z / fx
    Y = (vL - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float64)

# ================== ĐỌC CALIB & ẢNH (UNDISTORT) ==================
with open(CALIB_JSON, "r", encoding="utf-8") as f:
    calib = json.load(f)

K_orig      = np.array(calib["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32)
K_new       = np.array(calib["new_camera_matrix"], dtype=np.float32)

print("K_orig =\n", K_orig)
print("K_new  =\n", K_new)

imgL_raw = cv.imread(LEFT_IMG)
imgR_raw = cv.imread(RIGHT_IMG)
if imgL_raw is None or imgR_raw is None:
    raise SystemExit("Không đọc được ảnh LEFT/RIGHT, kiểm tra đường dẫn.")

h, w = imgL_raw.shape[:2]
assert imgR_raw.shape[:2] == (h, w), "Hai ảnh phải cùng kích thước!"

# Undistort về cùng ma trận nội K_new
imgL = cv.undistort(imgL_raw, K_orig, dist_coeffs, None, K_new)
imgR = cv.undistort(imgR_raw, K_orig, dist_coeffs, None, K_new)

grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

# Blur nhẹ cho KLT
grayL_b = cv.GaussianBlur(grayL, (3,3), 0)
grayR_b = cv.GaussianBlur(grayR, (3,3), 0)

# Ảnh hiển thị (resize)
imgL_vis = cv.resize(imgL, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
imgR_vis = cv.resize(imgR, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

# ================== THAM SỐ KLT ==================
lk_params = dict(
    winSize  = (WIN, WIN),
    maxLevel = MAX_LEVEL,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 1e-4),
    flags    = 0,
    minEigThreshold = 1e-4
)

# Lưu các điểm đã click (pixel + 3D)
clicked_pairs = []   # mỗi phần tử: {"uL","vL","uR","vR","P3D"}
point_index   = 0

# ================== CALLBACK CHUỘT ==================
def on_mouse_left(event, x, y, flags, param):
    global imgL_vis, imgR_vis, clicked_pairs, point_index

    if event == cv.EVENT_LBUTTONDOWN:
        # Toạ độ pixel full-res trên ảnh TRÁI
        uL = x / DISPLAY_SCALE
        vL = y / DISPLAY_SCALE

        print("\n======================================")
        print(f"Điểm mới #{point_index+1}")
        print(f"Pixel TRÁI (u_L, v_L) = ({uL:.2f}, {vL:.2f})")

        # Vẽ điểm trên ảnh trái
        cv.circle(imgL_vis, (x, y), 6, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(imgL_vis, str(point_index+1), (x+8, y-8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv.LINE_AA)
        cv.imshow("Left", imgL_vis)

        # Chuẩn bị p0 cho LK
        p0 = np.array([[[uL, vL]]], dtype=np.float32)

        # --- Forward LK: trái -> phải ---
        p1, st1, err1 = cv.calcOpticalFlowPyrLK(grayL_b, grayR_b, p0, None, **lk_params)
        if st1 is None or st1[0,0] == 0:
            print("KLT forward: status=0, không track được.")
            return

        # --- Backward LK: phải -> trái ---
        p0r, st2, err2 = cv.calcOpticalFlowPyrLK(grayR_b, grayL_b, p1, None, **lk_params)
        if st2 is None or st2[0,0] == 0:
            print("KLT backward: status=0, track không tin cậy.")
            return

        # Forward-backward error
        fb = np.linalg.norm(p0.reshape(-1,2) - p0r.reshape(-1,2), axis=1)[0]
        print(f"Forward-backward error = {fb:.3f} px")
        if fb > FB_THRESH:
            print(f"!!! FB error > {FB_THRESH} px → có thể match sai, bỏ.")
            return

        # Điểm tương ứng bên PHẢI
        uR, vR = p1.reshape(-1,2)[0]
        vert_disp = abs(vL - vR)
        print(f"Pixel PHẢI (u_R, v_R) = ({uR:.2f}, {vR:.2f})")
        print(f"Chênh lệch dọc |v_L - v_R| = {vert_disp:.2f} px")

        # **OPTION 2**: chỉ CẢNH BÁO, KHÔNG BỎ ĐIỂM
        if vert_disp > 5.0:
            print("!!! Cảnh báo: vertical disparity lớn, mô hình song song có thể sai,"
                  " nhưng vẫn dùng điểm này để tính 3D.")

        # Vẽ điểm trên ảnh phải
        xR_disp = int(round(uR * DISPLAY_SCALE))
        yR_disp = int(round(vR * DISPLAY_SCALE))
        cv.circle(imgR_vis, (xR_disp, yR_disp), 6, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(imgR_vis, str(point_index+1), (xR_disp+8, yR_disp-8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv.LINE_AA)
        cv.imshow("Right", imgR_vis)

        # Tính 3D point
        try:
            P3D = pixel_to_3d(uL, vL, uR, vR, K_new, BASELINE_MM)
        except ValueError as e:
            print("Lỗi tính 3D:", e)
            return

        d = uL - uR
        print(f"Disparity d = u_L - u_R = {d:.3f} px")
        print(f"P3D (X, Y, Z) = ({P3D[0]:.2f}, {P3D[1]:.2f}, {P3D[2]:.2f}) mm")

        # Lưu lại
        clicked_pairs.append({
            "uL": uL, "vL": vL,
            "uR": uR, "vR": vR,
            "P3D": P3D
        })
        point_index += 1

        # Nếu đã có >= 2 điểm, tính chiều dài đoạn nối 2 điểm cuối
        if len(clicked_pairs) >= 2:
            P1 = clicked_pairs[-2]["P3D"]
            P2 = clicked_pairs[-1]["P3D"]
            L = np.linalg.norm(P1 - P2)

            print("--------------------------------------")
            print("ĐOẠN NỐI 2 ĐIỂM GẦN NHẤT:")
            print(f"P1 3D = ({P1[0]:.2f}, {P1[1]:.2f}, {P1[2]:.2f}) mm")
            print(f"P2 3D = ({P2[0]:.2f}, {P2[1]:.2f}, {P2[2]:.2f}) mm")
            print(f"Chiều dài ước tính L = {L:.2f} mm")

            # Vẽ line nối 2 điểm cuối trên cả 2 ảnh
            uL1, vL1 = clicked_pairs[-2]["uL"], clicked_pairs[-2]["vL"]
            uL2, vL2 = clicked_pairs[-1]["uL"], clicked_pairs[-1]["vL"]
            uR1, vR1 = clicked_pairs[-2]["uR"], clicked_pairs[-2]["vR"]
            uR2, vR2 = clicked_pairs[-1]["uR"], clicked_pairs[-1]["vR"]

            ptL1 = (int(uL1*DISPLAY_SCALE), int(vL1*DISPLAY_SCALE))
            ptL2 = (int(uL2*DISPLAY_SCALE), int(vL2*DISPLAY_SCALE))
            ptR1 = (int(uR1*DISPLAY_SCALE), int(vR1*DISPLAY_SCALE))
            ptR2 = (int(uR2*DISPLAY_SCALE), int(vR2*DISPLAY_SCALE))

            cv.line(imgL_vis, ptL1, ptL2, (0,255,0), 2, cv.LINE_AA)
            cv.line(imgR_vis, ptR1, ptR2, (0,255,0), 2, cv.LINE_AA)
            cv.imshow("Left", imgL_vis)
            cv.imshow("Right", imgR_vis)

# ================== MAIN LOOP ==================
cv.namedWindow("Left",  cv.WINDOW_NORMAL)
cv.namedWindow("Right", cv.WINDOW_NORMAL)
cv.imshow("Left",  imgL_vis)
cv.imshow("Right", imgR_vis)

cv.setMouseCallback("Left", on_mouse_left)

print("HƯỚNG DẪN:")
print("- Ảnh 'Left' & 'Right' đã undistort dùng K_new.")
print(f"- Baseline giả sử = {BASELINE_MM} mm.")
print("- Click LẦN 1 trên ảnh 'Left' → KLT tìm điểm tương ứng bên phải,"
      " tính 3D P1.")
print("- Click LẦN 2 → tính P2 và chiều dài 3D L = |P1-P2| (mm).")
print("- Nếu click thêm, mỗi lần sẽ đo đoạn nối 2 điểm CUỐI CÙNG.")
print("- Độ lệch dọc lớn vẫn được chấp nhận (chỉ cảnh báo).")
print("- Nhấn ESC để thoát.\n")

while True:
    key = cv.waitKey(0) & 0xFF
    if key == 27:  # ESC
        break

cv.destroyAllWindows()