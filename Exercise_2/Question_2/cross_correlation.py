import cv2
import numpy as np

# Ma trận ảnh (6x7)
img = np.array([
    [1,  2,  3,  4,  9, 10, 11],
    [8,  9, 10, 11, 16, 17, 18],
    [15,16, 17, 18,  2,  3,  2],
    [1,  2,  3,  2,  9, 10, 11],
    [7,  9, 11,  9, 16, 17, 18],
    [15,15, 16, 16,  2,  3,  2]
], dtype=np.float32)

# Template (3x3)
templ = np.array([
    [9, 10, 11],
    [16,17, 18],
    [2,  3,  2]
], dtype=np.float32)

# Thực hiện Cross-Correlation 
result = cv2.matchTemplate(img, templ, method=cv2.TM_CCORR)

print("Kết quả Cross-Correlation:")
print(result)

# Tìm giá trị lớn nhất trong ma trận kết quả
max_val = np.max(result)
# Tìm tất cả các vị trí có giá trị bằng max_val
locations = np.argwhere(result == max_val)
print("Giá trị Cross-Correlation lớn nhất:", max_val)
print("Tất cả các vị trí (hàng, cột):")
for loc in locations:
    print(f"  {tuple(map(int, loc))}")  # Chuyển đổi numpy.int64 sang int
