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
corner_img = cv.imread('corner_edge.png', cv.IMREAD_GRAYSCALE)
print("Kích thước ảnh đỉnh mẫu ban đầu:", corner_img.shape)
# resize_corner = cv.resize(corner_img, corner_resize)

# Làm mượt
# img_blurred = cv.GaussianBlur(resize_img, (11, 11), 0)
# corner_blurred = cv.GaussianBlur(resize_corner, (5, 5), 0)

# Threshold (ra dạng 0–1)
_, binary_img = cv.threshold(resize_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
_, binary_corner = cv.threshold(corner_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


edges = cv.Canny(binary_img, 30, 150)


# Kernel
kernel = binary_corner.copy()

# Tạo bản sao để floodfill
im_floodfill = kernel.copy()

# Tạo mask (phải lớn hơn ảnh 2 pixel theo mỗi chiều)
h, w = kernel.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
print(mask)
# Flood fill từ góc ảnh (giả định vùng nền ngoài cùng)
cv.floodFill(im_floodfill, mask, (0, 30), 255)
print("Intensity tại (x, y): ", im_floodfill[10, 30])
im_floodfill = im_floodfill // 255

plt.figure(figsize=(12, 6))
plt.imshow(im_floodfill, cmap='gray')
plt.title('Ảnh nhị phân góc đỉnh ngôi sao')
plt.show()

count_1 = np.sum(im_floodfill == 0)
print("Số phần tử bằng 1 trong im_floodfill là:", count_1)



kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
print(kernel_erode)
kernel_hit_miss_erode = cv.morphologyEx(im_floodfill, cv.MORPH_ERODE, kernel_erode, iterations = 2)
kernel_hit_miss_dilate = cv.morphologyEx(im_floodfill, cv.MORPH_DILATE, kernel_erode, iterations = 2)





kernel_hit = im_floodfill.copy()
kernel_miss = np.where(im_floodfill == 0, -1, 0)
kernel_misshit = kernel_hit + kernel_miss


subtract_1 = cv.subtract(im_floodfill, kernel_hit_miss_erode, dtype = cv.CV_8U)
subtract_2 = cv.subtract(kernel_hit_miss_dilate, im_floodfill, dtype = cv.CV_8U)
result = cv.add(subtract_1, subtract_2)

# Khởi tạo kernel mới toàn -1 (background mặc định)
kernel_new = np.full(kernel_hit_miss_erode.shape, -1, dtype=np.int8)

# Nếu pixel trong erosion = 1 ⇒ foreground
kernel_new[kernel_hit_miss_erode == 1] = 1

# Nếu pixel trong result = 1 ⇒ don't care
kernel_new[result == 1] = 0

# Nếu pixel trong dilation = 0 ⇒ background (-1)
kernel_new[kernel_hit_miss_dilate == 0] = -1


count_2 = np.sum(kernel_misshit == -1)
print("Số phần tử bằng 1 trong im_floodfill là:", count_2)


count_zero = np.sum(kernel_misshit == 0)
print("Số phần tử bằng 0 là:", count_zero)
print("Kích thước kernel:", kernel_misshit.shape)
print("Kernel sinh ra:\n", kernel_misshit)

output_image = cv.morphologyEx(binary_img, cv.MORPH_HITMISS, kernel_new.astype(np.int8))

# -----------------------------
# B4. Hiển thị kết quả
# -----------------------------


plt.figure(figsize=(18, 10)) 
plt.subplot(2, 3, 1)
plt.imshow(subtract_1, cmap='gray')
plt.title('Ảnh nhị phân góc đỉnh ngôi sao')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(subtract_2, cmap='gray')
plt.title('Kết quả phát hiện đỉnh (Hit-or-Miss)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(result, cmap='gray')   # ảnh thứ ba
plt.title('Ảnh khác hoặc kết quả trung gian')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(kernel_hit_miss_erode, cmap='gray')
plt.title('Ảnh nhị phân góc đỉnh ngôi sao')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(kernel_hit_miss_dilate, cmap='gray')
plt.title('Kết quả phát hiện đỉnh (Hit-or-Miss)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(im_floodfill, cmap='gray')   # ảnh thứ ba
plt.title('Ảnh khác hoặc kết quả trung gian')
plt.axis('off')



# -----------------------------
# B5. Đánh dấu vị trí đỉnh lên ảnh
# -----------------------------
coords = np.column_stack(np.where(output_image > 0))
print("Số đỉnh phát hiện được:", len(coords))
print("Toạ độ các đỉnh:\n", coords)

img_color = cv.cvtColor((binary_img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
for (y, x) in coords:
    cv.circle(img_color, (x, y), 10, (0, 0, 255), -1)
# plt.figure(figsize=(12, 6))

plt.figure(figsize=(8, 8))

# Ảnh nền: ảnh nhị phân
plt.imshow(binary_img, cmap='gray')

# Ảnh màu có đánh dấu: đè lên (phải chuyển sang RGB để matplotlib hiểu)
plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), alpha=0.3)

plt.title('Ảnh nhị phân và vị trí đỉnh phát hiện được')
plt.axis('off')
plt.show()


cv.imshow("Binary", binary_img)
cv.waitKey()
cv.destroyAllWindows()
