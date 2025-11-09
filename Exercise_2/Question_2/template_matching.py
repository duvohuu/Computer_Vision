import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return [] 
    boxes = np.array(boxes)  
    # Lấy tọa độ các góc
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = boxes[:, 4]
    # Tính diện tích
    areas = (x2 - x1) * (y2 - y1)
    # Sắp xếp theo score giảm dần
    idxs = np.argsort(scores)[::-1]
    pick = []
    while len(idxs) > 0:
        # Lấy box có score cao nhất
        i = idxs[0]
        pick.append(i) 
        # Tính IoU với các box còn lại
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs[1:]]
        # Loại bỏ các box có overlap cao
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))
    return boxes[pick]

# --- Đọc ảnh gốc ---
img_color = cv2.imread("source_image.png")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# --- Danh sách các template và threshold ---
templates = [
    {"path": "template_image.png",   "threshold": 0.56, "color": (0, 255, 0)},    # Xanh lá
    {"path": "template_image_2.png", "threshold": 0.75, "color": (255, 0, 0)},    # Xanh dương
    {"path": "template_image_3.png", "threshold": 0.74, "color": (0, 0, 255)}     # Đỏ
]

# --- Vòng lặp qua từng template ---
for t in templates:
    template = cv2.imread(t["path"], cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Không thể đọc được {t['path']}. Kiểm tra lại đường dẫn!")
        continue
    h, w = template.shape
    # Thực hiện template matching
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # Tìm tất cả vị trí có độ khớp cao hơn ngưỡng
    locations = np.where(result >= t["threshold"])
    if locations[0].size == 0:
        print(f"Không tìm thấy mẫu {t['path']} (threshold={t['threshold']})")
        continue
    # Tạo danh sách các bounding box kèm score
    boxes = []
    for pt in zip(*locations[::-1]):
        score = result[pt[1], pt[0]]
        boxes.append([pt[0], pt[1], w, h, score])
    print(f"Tìm thấy {len(boxes)} vị trí trước xử lý cho {t['path']}") 
    # Xử lý các vị trí bị chồng chéo
    boxes_nms = non_max_suppression(boxes, overlapThresh=0.3)
    # Vẽ hình chữ nhật cho từng vị trí sau NMS
    for box in boxes_nms:
        x, y, w_box, h_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img_color, (x, y), (x + w_box, y + h_box), t["color"], 2)
    print(f"Đã tìm thấy {len(boxes_nms)} vị trí sau xử lý cho {t['path']} (threshold={t['threshold']})")

# --- Hiển thị kết quả ---
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Kết quả Template Matching nhiều mẫu")
plt.axis("off")
plt.show()
