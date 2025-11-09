import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Đọc hai ảnh ---S
imgA = cv2.imread("/run/media/vhdu/WORK/HCMUT/HK251/Computer Vision/Exercises/Exercise_2/Question_1/images/book1.png")
imgB = cv2.imread("/run/media/vhdu/WORK/HCMUT/HK251/Computer Vision/Exercises/Exercise_2/Question_1/images/book2.png")

if imgA is None or imgB is None:
    raise FileNotFoundError("Không tìm thấy ảnh book1.png hoặc book2.png trong thư mục image/")

# --- Chuyển sang ảnh xám ---
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# --- Đảm bảo 2 ảnh cùng kích thước ---
if grayA.shape != grayB.shape:
    grayB = cv2.resize(grayB, (grayA.shape[1], grayA.shape[0]))
    imgB = cv2.resize(imgB, (grayA.shape[1], grayA.shape[0]))
    print("Ảnh B đã được resize để khớp kích thước với ảnh A.")

# ==========================
# 1️⃣ HARRIS CORNER DETECTION (ảnh A)
# ==========================
blockSize = 3
ksize = 3
k = 0.04

grayA_f = np.float32(grayA)
R = cv2.cornerHarris(grayA_f, blockSize, ksize, k)

# Chuẩn hóa kết quả về 0-255
R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
R_norm = np.uint8(R_norm)

# --- Lấy 4 điểm có giá trị R lớn nhất (4 góc mạnh nhất) ---
R_flat = R.flatten()
top4_idx = np.argsort(R_flat)[-4:]
h, w = R.shape
points = np.array([[int(idx % w), int(idx // w)] for idx in top4_idx], dtype=np.float32)
points = points.reshape(-1, 1, 2)

print("Tọa độ 4 góc tìm được (Harris):")
print(points.reshape(-1, 2))

# --- Vẽ 4 góc tìm được trên ảnh A ---
imgA_draw = imgA.copy()
for i, p in enumerate(points):
    x, y = p.ravel()
    cv2.circle(imgA_draw, (int(x), int(y)), 6, (0, 0, 255), -1)
    cv2.putText(imgA_draw, f"P{i+1}", (int(x)+8, int(y)-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ==========================
# 2️⃣ THEO DÕI BẰNG KLT (ảnh B)
# ==========================
corners_next, status, err = cv2.calcOpticalFlowPyrLK(
    grayA, grayB,
    points, None,
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Chỉ lấy các điểm track thành công
good_old = points[status == 1]
good_new = corners_next[status == 1]

print("Vị trí 4 góc sau khi track (KLT):")
print(good_new.reshape(-1, 2))

# ==========================
# 3️⃣ VẼ KẾT QUẢ
# ==========================
imgB_draw = imgB.copy()

for i, (new, old) in enumerate(zip(good_new, good_old)):
    x_new, y_new = new.ravel()
    x_old, y_old = old.ravel()

    # Vẽ điểm mới
    cv2.circle(imgB_draw, (int(x_new), int(y_new)), 6, (255, 0, 0), -1)
    cv2.putText(imgB_draw, f"P{i+1}", (int(x_new)+8, int(y_new)-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    # Vẽ vector chuyển động
    cv2.arrowedLine(imgB_draw, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0,255,0), 2, tipLength=0.2)

# ==========================
# 4️⃣ HIỂN THỊ KẾT QUẢ
# ==========================
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(imgA_draw, cv2.COLOR_BGR2RGB))
plt.title("Ảnh A - 4 góc phát hiện bằng Harris")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(imgB_draw, cv2.COLOR_BGR2RGB))
plt.title("Ảnh B - Theo dõi 4 góc bằng KLT")
plt.axis("off")

plt.tight_layout()
plt.show()
