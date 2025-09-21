import cv2
import numpy as np
import math

# Danh sách bán kính
radius_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]

# Kích thước ảnh sau khi resize
target_size = (100, 100)

# Danh sách đường dẫn ảnh
images = ["ex4_1.png", "ex4_2.png", "ex4_3.png", "ex4_4.png"]

# Biến lưu số pixel và số pixel trắng trên mỗi bán kính
white_pixels = np.zeros((len(images), len(radius_list)), dtype = int)

# a) Đọc số pixel trắng trên mỗi bán kính và ghi ra file result.txt
with open('result.txt', 'w') as f:
     for i, image_path in enumerate(images):
          f.write(f"Image {i + 1}: {image_path}\n")
          img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load ảnh gray
          if img is None:
               print(f"Không thể load được {image_path}")
               continue
          # Resize lại ảnh
          img_resize = cv2.resize(img, target_size)

          # Xác định tâm vòng tròn
          center_x, center_y = target_size[0] // 2, target_size[1] // 2
          
          for j, r in enumerate(radius_list):
               count = 0
               for theta in np.arange(0, 2*np.pi, 0.05):
                    x = int(center_x + r * np.cos(theta))
                    y = int(center_y + r * np.sin(theta))
                    
                    # Kiểm tra ngưỡng: 128
                    if (0 < x < target_size[0] and 0 < y < target_size[1]):
                         if img_resize[y, x] > 128:
                              count += 1
               
               white_pixels[i][j] = count               
               f.write(f"Bán kính: {r}, số pixel trắng: {count}\n")
          f.write("\n")
          
# So sánh 2 ảnh image1 và image2

# b)
# So sánh ảnh số 2 và số 4 (mong đợi kết quả là 2 hình giống nhau)
image1, image2 = 2, 3
vector_1 = white_pixels[image1 - 1]
vector_2 = white_pixels[image2 - 1]

# Tính trung bình các vector đặc trưng của mỗi hình
mean1 = np.mean(vector_1)
mean2 = np.mean(vector_2)

# Tính tử số của công thức NCC (Normalized Cross-Correlation)
num = np.sum((vector_1 - mean1) * (vector_2 - mean2))

# Tính mẫu số
den = np.sqrt(np.sum((vector_1 - mean1) ** 2)) * np.sqrt(np.sum((vector_2 - mean2) ** 2))



if den == 0:
    print("Error NaN")
else:
    corr = num / den
    print(f"Tỉ lệ giống nhau của ảnh {image1} ảnh {image2}: {corr:.3f}")
    if corr >= 0.9:
        print(f"Hai ảnh {image1} và {image2} giống nhau")
    else:
        print(f"Hai ảnh {image1} và {image2} không giống nhau")


                    
               
          

         
          