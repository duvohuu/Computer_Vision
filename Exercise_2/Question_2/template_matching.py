import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Đọc ảnh gốc ---
img_color = cv2.imread("source_image.png")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# --- Danh sách các template và threshold ---
templates = [
    {"path": "template_image.png",   "threshold": 0.56, "color": (0, 255, 0)},    # Xanh lá
    {"path": "template_image_2.png", "threshold": 0.75, "color": (255, 0, 0)},    # Xanh dương
    {"path": "template_image_3.png", "threshold": 0.74, "color": (0, 0, 255)}   # Đỏ
]

# --- Vòng lặp qua từng template ---
for t in templates:
    template = cv2.imread(t["path"], cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"⚠️ Không thể đọc được {t['path']}. Kiểm tra lại đường dẫn!")
        continue

    h, w = template.shape

    # Thực hiện template matching
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # Tìm tất cả vị trí có độ khớp cao hơn ngưỡng
    locations = np.where(result >= t["threshold"])

    # Nếu không tìm thấy kết quả
    if locations[0].size == 0:
        print(f"❌ Không tìm thấy mẫu {t['path']} (threshold={t['threshold']})")
        continue

    # Vẽ hình chữ nhật cho từng vị trí khớp
    for pt in zip(*locations[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img_color, pt, bottom_right, t["color"], 2)

    print(f"✅ Đã tìm thấy {len(locations[0])} vị trí cho {t['path']} (threshold={t['threshold']})")

# --- Hiển thị kết quả ---
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Kết quả Template Matching nhiều mẫu")
plt.axis("off")
plt.show()
