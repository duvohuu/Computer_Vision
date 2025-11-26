import cv2
import numpy as np
import glob

# ==================================================
# 1. TẠO R-TABLE TỪ TEMPLATE
# ==================================================
def build_r_table(template):
    edges = cv2.Canny(template, 80, 150)
    gy, gx = np.gradient(template.astype(np.float32))
    angle = np.arctan2(gy, gx)

    ys, xs = np.where(edges > 0)

    cy, cx = np.mean(ys), np.mean(xs)

    R = {}
    angle_bin = 20  # chia 20°/bin

    for (y, x) in zip(ys, xs):
        theta = int((angle[y, x] * 180 / np.pi) // angle_bin)
        if theta not in R:
            R[theta] = []
        r = np.array([cy - y, cx - x])   # vector từ điểm biên → tâm
        R[theta].append(r)

    return R


# ==================================================
# 2. GENERALIZED HOUGH TRANSFORM – TỰ LÀM
# ==================================================
def generalized_hough(scene, R):
    edges = cv2.Canny(scene, 80, 150)
    gy, gx = np.gradient(scene.astype(np.float32))
    angle = np.arctan2(gy, gx)

    H = np.zeros_like(scene, dtype=np.float32)

    ys, xs = np.where(edges > 0)
    angle_bin = 20

    for (y, x) in zip(ys, xs):
        theta = int((angle[y, x] * 180 / np.pi) // angle_bin)
        if theta in R:
            for r in R[theta]:
                yc = int(y + r[0])
                xc = int(x + r[1])
                if 0 <= yc < H.shape[0] and 0 <= xc < H.shape[1]:
                    H[yc, xc] += 1

    return H


# ==================================================
# 3. NON-MAX SUPPRESSION ĐỂ LẤY TẤT CẢ MÁY BAY
# ==================================================
def find_peaks(H, threshold_ratio=0.5, radius=15):
    H_copy = H.copy()
    peaks = []

    threshold = threshold_ratio * H.max()

    for y in range(H.shape[0]):
        for x in range(H.shape[1]):
            if H_copy[y, x] >= threshold:
                # check local maxima
                y1 = max(0, y - radius)
                y2 = min(H.shape[0], y + radius)
                x1 = max(0, x - radius)
                x2 = min(H.shape[1], x + radius)

                window = H_copy[y1:y2, x1:x2]
                if H_copy[y, x] == window.max():
                    peaks.append((x, y))
                    H_copy[y1:y2, x1:x2] = 0  # remove neighborhood

    return peaks


# ==================================================
# 4. CHẠY PHẦN A — GHT MULTI-DETECTION
# ==================================================
scene = cv2.imread("pictures/aircraft_scene.png", 0)
scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

template_paths = sorted(glob.glob("pictures/template_*.png"))

all_peaks = []

for tpath in template_paths:
    template = cv2.imread(tpath, 0)
    R = build_r_table(template)
    H = generalized_hough(scene, R)
    peaks = find_peaks(H, threshold_ratio=0.45, radius=20)

    print(f"{tpath} → Detect {len(peaks)} máy bay")

    for p in peaks:
        all_peaks.append(p)

# Vẽ đánh số
for i, (x, y) in enumerate(all_peaks, 1):
    cv2.circle(scene_color, (x, y), 12, (0, 0, 255), 2)
    cv2.putText(scene_color, str(i), (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("GHT Detection", scene_color)
cv2.imwrite("GHT_detected.png", scene_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
