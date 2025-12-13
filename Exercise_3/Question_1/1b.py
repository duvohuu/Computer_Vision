import cv2
import numpy as np
import math

VIDEO_PATH = r"D:\HK251\vision\HW\HW3\Bai1_b\56310-479197605_large.mp4"

# ================== THAM SỐ CẦN TUNING ==================
# Lọc blob (contour)
MIN_AREA      = 800       # diện tích tối thiểu coi là xe
MAX_AREA      = 50000     # diện tích tối đa (quá to có thể là nhiễu)
MIN_W, MIN_H  = 15, 15    # kích thước tối thiểu bounding box
MIN_RATIO     = 0.4       # tỉ lệ w/h thấp nhất (quá đứng loại)
MAX_RATIO     = 3.0       # tỉ lệ w/h cao nhất (quá ngang loại)

# Đếm qua line
DIST_THRESH   = 10        # khoảng cách tối đa từ TÂM xe tới line (px)
MIN_FRAME_GAP = 6         # số frame tối thiểu giữa 2 lần đếm trên cùng line

# Morphology
KERNEL_OPEN_SIZE  = (3, 3)   # xoá nhiễu nhỏ
KERNEL_CLOSE_SIZE = (6, 6)   # nối các mảng, lấp lỗ

# ================== CẤU TRÚC LƯU LINE ==================
lines = []
drawing_line = False
temp_p1 = None

# ================== HÀM PHỤ ==================
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Khoảng cách từ điểm P tới đoạn thẳng P1P2."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1

    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)

    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2)

    b = c1 / c2
    bx = x1 + b * vx
    by = y1 + b * vy
    return math.hypot(px - bx, py - by)

def mouse_callback(event, x, y, flags, param):
    """Click chuột trái 2 lần để vẽ một đường đếm."""
    global drawing_line, temp_p1, lines
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing_line:
            drawing_line = True
            temp_p1 = (x, y)
        else:
            drawing_line = False
            p2 = (x, y)
            if math.hypot(p2[0] - temp_p1[0], p2[1] - temp_p1[1]) > 5:
                lines.append({
                    'p1': temp_p1,
                    'p2': p2,
                    'count': 0,
                    'last_count_frame': -999999
                })
            temp_p1 = None

def is_valid_blob(cnt):
    """Lọc blob theo diện tích / kích thước / tỉ lệ w/h."""
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    if w < MIN_W or h < MIN_H:
        return False

    ratio = float(w) / float(h)
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return False

    return True

# ================== 1. MỞ VIDEO & LẤY FRAME ĐẦU VẼ LINE ==================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Không mở được video.")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Không đọc được frame đầu.")
    exit()

first_frame = cv2.resize(first_frame, (960, 540))
h, w = first_frame.shape[:2]

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

print("Dùng chuột trái click 2 điểm để vẽ line đếm (muốn nhiều line thì vẽ nhiều).")
print("Nhấn phím 's' để bắt đầu chạy video, ESC để thoát.")

while True:
    display = first_frame.copy()

    for idx, line in enumerate(lines):
        x1, y1 = line['p1']
        x2, y2 = line['p2']
        cv2.line(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(display, f"{idx+1}: {line['count']}",
                    (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if drawing_line and temp_p1 is not None:
        cv2.circle(display, temp_p1, 4, (0, 255, 255), -1)

    cv2.putText(display, "Draw lines, then press 's' to start",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    cv2.imshow("Frame", display)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    if key == ord('s'):
        break

cap.release()
cap = cv2.VideoCapture(VIDEO_PATH)

# ================== BACKGROUND SUBTRACTOR & KERNEL ==================
backSub = cv2.createBackgroundSubtractorMOG2(
    history=450, varThreshold=17, detectShadows=True
)

kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_OPEN_SIZE)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_CLOSE_SIZE)

frame_idx = 0

# ================== 2. VÒNG LẶP CHẠY VIDEO & ĐẾM XE ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame = cv2.resize(frame, (960, 540))

    # ---------- 2.1. LẤY FG MASK ----------
    fgMask = backSub.apply(frame)

    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

    # ---------- 2.2. MORPHOLOGY (OPEN + CLOSE) ----------
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN,
                              kernel_open, iterations=1)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE,
                              kernel_close, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel_close, iterations=1)

    # ---------- 2.3. TÌM CONTOUR (BLOB) ----------
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if not is_valid_blob(cnt):
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)

        cx = x + w_box // 2
        cy = y + h_box // 2

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box),
                      (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        for line in lines:
            x1, y1 = line['p1']
            x2, y2 = line['p2']

            dist = point_to_segment_distance(cx, cy, x1, y1, x2, y2)

            if dist < DIST_THRESH:
                if frame_idx - line['last_count_frame'] >= MIN_FRAME_GAP:
                    line['count'] += 1
                    line['last_count_frame'] = frame_idx

    # ---------- 2.5. VẼ LẠI LINE + COUNT ----------
    for idx, line in enumerate(lines):
        x1, y1 = line['p1']
        x2, y2 = line['p2']
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(frame, f"{idx+1}: {line['count']}",
                    (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground", fgMask)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
