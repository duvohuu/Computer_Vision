import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# B1. Đọc và xử lý ảnh
# -----------------------------
img_resize = (1000, 1000)
corner_resize = (50, 50)

# Ảnh ngôi sao
img = cv.imread('ex3.png', cv.IMREAD_GRAYSCALE)
resize_img = cv.resize(img, img_resize)
 
# Ảnh đỉnh mẫu
corner_img = cv.imread('ex3_corner_1.png', cv.IMREAD_GRAYSCALE)
print("Kích thước ảnh đỉnh mẫu ban đầu:", corner_img.shape)
# resize_corner = cv.resize(corner_img, corner_resize)

# Làm mượt
img_blurred = cv.GaussianBlur(resize_img, (11, 11), 0)
corner_blurred = cv.GaussianBlur(corner_img, (5, 5), 0)


# Threshold (ra dạng 0–1)
_, binary_img = cv.threshold(img_blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
_, binary_corner = cv.threshold(corner_blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

binary_corner = binary_corner // 255

# Plot xem thử ảnh nhị phân ở đỉnh
plt.figure(figsize=(12, 6))
plt.imshow(binary_corner, cmap='gray')
plt.title('Ảnh nhị phân góc đỉnh ngôi sao')
plt.show()

# Lưu ảnh nhị phân góc ở đỉnh ngôi sao thành file trong thư mục hiện tại
cv.imwrite('binary_corner.png', (binary_corner * 255).astype(np.uint8))

# -----------------------------
# B2. Tạo kernel cho Hit-or-Miss và thực hiện phép biến đổi
# -----------------------------


# Tạo ra các vùng để tạo kernel cho hit miss
kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
erode = cv.morphologyEx(binary_corner, cv.MORPH_ERODE, kernel_erode, iterations = 2)
dilate = cv.morphologyEx(binary_corner, cv.MORPH_DILATE, kernel_erode, iterations = 2)

# Tạo ra vùng don't care trong kernel hit-miss
subtract_1 = cv.subtract(binary_corner, erode, dtype = cv.CV_8U)
subtract_2 = cv.subtract(dilate, binary_corner, dtype = cv.CV_8U)
dont_care_region = cv.add(subtract_1, subtract_2)

# Khởi tạo kernel mới toàn -1 (background mặc định)
kernel_new = np.full(erode.shape, -1, dtype=np.int8)

# Ảnh erode được chọn làm vùng foreground trong kernel hit-miss
kernel_new[erode == 1] = 1

# Vùng don't care được chọn làm vùng không quan tâm trong kernel hit-miss để tránh quá sát
kernel_new[dont_care_region == 1] = 0

# Vùng bù ảnh ảnh dilate được chọn làm vùng background chắc chắn
kernel_new[dilate == 0] = -1


# Giải thuật Hit-or-Miss
output_image = cv.morphologyEx(binary_img, cv.MORPH_HITMISS, kernel_new.astype(np.int8))

# -----------------------------
# B3. Hiển thị kết quả
# -----------------------------


plt.figure(figsize=(25, 25)) 
plt.subplot(2, 3, 1)
plt.imshow(binary_corner, cmap='gray')
plt.title('Ảnh nhị phân góc đỉnh ngôi sao')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(erode, cmap='gray')
plt.title('Kết quả erode')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilate, cmap='gray')   # ảnh thứ ba
plt.title('Kết quả dilate')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(subtract_1, cmap='gray')
plt.title('Kết quả trừ (Erode - Binary Corner)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(subtract_2, cmap='gray')
plt.title('Kết quả trừ (Dilate - Binary Corner)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(dont_care_region, cmap='gray')   # ảnh thứ ba
plt.title('Vùng không quan tâm trong kernel hit-or-miss')
plt.axis('off')

# Lưu figure này thành 1 ảnh trong thư mục hiện tại
plt.savefig('hit_miss_intermediate_steps.png')


# -----------------------------
# B4. Đánh dấu vị trí đỉnh lên ảnh
# -----------------------------
coords = np.column_stack(np.where(output_image > 0))
print("Số đỉnh phát hiện được:", len(coords))
print("Toạ độ các đỉnh:\n", coords)

img_color = cv.cvtColor((binary_img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
for (y, x) in coords:
    cv.circle(img_color, (x, y), 10, (0, 0, 255), -1)

plt.figure(figsize=(16, 16))

# Ảnh nền: ảnh nhị phân
plt.imshow(binary_img, cmap='gray')
plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), alpha=0.7)
# Lưu ảnh này thành file trong thư mục hiện tại
plt.savefig('detected_corners.png')

plt.title('Ảnh nhị phân và vị trí đỉnh phát hiện được')
plt.axis('off')
plt.show()


