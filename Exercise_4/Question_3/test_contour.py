import cv2
import numpy as np

# Load ảnh
img = cv2.imread(r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\test.png")

# Chuyển sang HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Giới hạn màu xanh (blue) – tùy chỉnh theo ảnh thật
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Mask theo màu
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Tìm contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lấy contour lớn nhất
c = max(contours, key=cv2.contourArea)

# Vẽ contour lên ảnh
img_contour = img.copy()
cv2.drawContours(img_contour, [c], -1, (0, 0, 255), 2)

cv2.imshow("Mask", mask)
cv2.imshow("Contour", img_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
