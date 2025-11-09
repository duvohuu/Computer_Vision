import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

resize = (1000, 1000)

# Đọc ảnh xám
img = cv.imread('ex3.png', cv.IMREAD_GRAYSCALE)
resize_img = cv.resize(img, resize)

# Làm mượt ảnh bằng Gaussian filter
blurred = cv.GaussianBlur(resize_img, (11, 11), 0)


# Threshold ra ảnh nhị phân
_, binary_img = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Loại bỏ nhiễu nhỏ bằng phép co-dãn (morphology)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
opened = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel, iterations = 2)
closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations = 2)


# Giữ lại blob lớn nhất (ngôi sao)
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(closed, connectivity=8)
# tìm label có diện tích lớn nhất (bỏ label nền)
largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
mask = np.uint8(labels == largest_label) * 255


# Tạo bản sao để floodfill
im_floodfill = mask.copy()

# Tạo mask (phải lớn hơn ảnh 2 pixel theo mỗi chiều)
h, w = mask.shape[:2]
mask_1 = np.zeros((h+2, w+2), np.uint8)
print(mask_1)
# Flood fill từ góc ảnh (giả định vùng nền ngoài cùng)
cv.floodFill(im_floodfill, mask_1, (100, 500), 255)
print("Intensity tại (x, y): ", im_floodfill[100, 500])
# cv.floodFill(im_floodfill, mask_1, (100, 500), 255)

# Đảo ảnh floodfill để được vùng "hole"
im_floodfill_inv = cv.bitwise_not(im_floodfill)
 
# Lấp lỗ = ảnh gốc OR với ảnh lỗ đã đảo
im_filled = im_floodfill | im_floodfill_inv


# ### Bước thêm: Flood fill để lấy vùng bên trong ngôi sao (vùng tam giác)
# tmp = im_floodfill.copy()
# cv.floodFill(tmp, mask_1, (0, 0), 255)

# # mask lúc này có vùng ngoài bị đánh dấu 1, bên trong ngôi sao là 0
# mask_1 = cv.bitwise_not(mask_1)

# # Bây giờ flood fill bên trong tam giác (ví dụ chọn seed ở khoảng giữa)
# im_floodfill_final = im_floodfill.copy()
# seed_point = (130, 500)  # ní chọn toạ độ nằm trong vùng đen của tam giác
# cv.floodFill(im_floodfill_final, mask_1, seed_point, 255)




kernel_1 = cv.getStructuringElement(cv.MORPH_DIAMOND, (13, 13))
closed_3 = cv.morphologyEx(im_floodfill_inv, cv.MORPH_CLOSE, kernel_1.T, iterations = 2)



print(kernel_1)


# Canny edge
edges = cv.Canny(mask, 50, 150)

cv.imshow("Original Image", resize_img)
# cv.imshow("im_floodfill_inverse final", im_floodfill_final)
cv.imshow("im_floodfill", im_floodfill)
cv.imshow("im_floodfill_inverse", im_floodfill_inv)
cv.imshow("Canny edges", edges)
cv.imshow("Opened after floodfill and opended", closed_3)
cv.waitKey()
cv.destroyAllWindows()


# Hiển thị kết quả
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title("Ảnh gốc")
plt.imshow(resize_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Sau Gaussian")
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Nhị phân + Morphology")
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Blob lớn nhất (ngôi sao)")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()










































# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt

# resize = (1000, 1000)

# # Read gray image
# img = cv.imread('ex3.png', cv.IMREAD_GRAYSCALE)

# resize_img = cv.resize(img, resize)

# # Làm mượt ảnh bằng Gaussian filter
# blurred = cv.GaussianBlur(resize_img, (10, 10), 0)   # kernel 5x5, sigma=0

# # Threshold để ra ảnh nhị phân (đảo ngược nền)
# _, binary_img = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


# # Kernel (cross)
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))
# kernel_0 = cv.getStructuringElement(cv.MORPH_RECT, (1, 10))
# kernel_1 = np.array([[0, 0, 0, 1],
#                     [0, 0, 1, 0],
#                     [0, 1, 0, 0],
#                     [1, 0, 0, 0]], np.uint8)
# kernel_2 = np.eye(5, dtype = np.uint8)

# # Opening
# opended = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel.T, iterations= 2)

# # opended_1 = cv.morphologyEx(opended, cv.MORPH_OPEN, kernel, iterations= 2)

# # Closing to restored 

# restored_img = cv.morphologyEx(opended, cv.MORPH_CLOSE, kernel, iterations= 2)

# # loại bỏ nhiêu thẳn
# opended_1 = cv.morphologyEx(restored_img, cv.MORPH_OPEN, kernel_0.T, iterations=2)
# #restored
# restored_img_1 = cv.morphologyEx(opended_1, cv.MORPH_CLOSE, kernel_0, iterations=2)



# cv.imshow("Original image", img)
# cv.imshow("Gaussian image", binary_img)
# cv.imshow("Opended image", opended)
# # cv.imshow("Opended image 1", opended_1)
# cv.imshow("Restored image", restored_img)
# cv.imshow("Opended image 2", opended_1)
# # cv.imshow("Opended image 1", opended_1)
# cv.imshow("Restored image 2", restored_img_1)
# cv.waitKey()
# cv.destroyAllWindows()