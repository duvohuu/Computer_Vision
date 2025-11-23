import cv2
import numpy as np
import math

# ================== CẤU HÌNH ==================
IMAGE_PATH = r"D:\HK251\vision\BTL\Bai_2\pictures\img2.jpg"   # đổi thành ảnh của bạn
CANNY_T1   = 50
CANNY_T2   = 150
KERNEL_SZ  = 3            # kernel cho opening/closing


def main():
    # -------- BƯỚC 1: TIỀN XỬ LÝ + PHÂN NGƯỠNG + CANNY --------
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError("Không đọc được ảnh, kiểm tra lại IMAGE_PATH")
    # GIỮ NGUYÊN KÍCH THƯỚC ẢNH
    img_draw = img.copy()
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    # Card sáng trên nền tối → THRESH_BINARY + OTSU
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Lọc nhiễu: opening (xóa hạt nhỏ) + closing (lấp lỗ nhỏ)
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    thresh_open  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    thresh_clean = cv2.morphologyEx(thresh_open, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Canny (chỉ để xem biên, không dùng để chọn contour chính)
    edges = cv2.Canny(thresh_clean, CANNY_T1, CANNY_T2)
    # -------- BƯỚC 2: TÌM CONTOUR & CHỌN CONTOUR LỚN NHẤT --------
    # Lấy contour trực tiếp từ ảnh nhị phân đã lọc
    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("Không tìm được contour.")
        return
    largest_cnt = max(contours, key=cv2.contourArea)
    # Vẽ contour lớn nhất (card)
    #cv2.drawContours(img_draw, [largest_cnt], -1, (0, 0, 255), 2)

    # -------- BƯỚC 3: TỌA ĐỘ TRỌNG TÂM & GÓC XOAY (KHÔNG PCA) --------
    # Trọng tâm
    M = cv2.moments(largest_cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    # Hình chữ nhật bao xoay
    rect = cv2.minAreaRect(largest_cnt)   # ((x,y), (w,h), angle)
    box  = cv2.boxPoints(rect)
    box  = np.int32(box)
    cv2.drawContours(img_draw, [box], 0, (0, 255, 0), 2)
    # Lấy cạnh dài nhất để tính góc
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

    # Góc có dấu so với trục Ox
    angle_rad = math.atan2(dy_best, dx_best)
    angle_deg = math.degrees(angle_rad)

    # Thu về (-90, 90] để dễ nhìn
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    print("Tọa độ trọng tâm (cx, cy):", cx, cy)
    print("Góc xoay (có dấu):", angle_deg, "độ")

    # Vẽ trọng tâm + hướng cạnh dài
    cv2.circle(img_draw, (cx, cy), 5, (255, 0, 0), -1)
    L = 100
    x2 = int(cx + L * (dx_best / max_len))
    y2 = int(cy + L * (dy_best / max_len))
    #cv2.line(img_draw, (cx, cy), (x2, y2), (0, 255, 255), 2)

    text = f"Center: ({cx},{cy}) Angle: {angle_deg:.2f} deg"
    cv2.putText(img_draw, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -------- BƯỚC 4: HIỂN THỊ & LƯU ẢNH --------
    cv2.imshow("Thresh_clean", thresh_clean)
    cv2.imshow("Edges", edges)
    cv2.imshow("Result", img_draw)

    cv2.imwrite("result_with_contour_center_angle.png", img_draw)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
