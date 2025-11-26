import cv2
import numpy as np

img = np.array([
    [0,   0,   0,   0,   0,   0,   0],
    [0,   0,   0, 255,   0,   0,   0],
    [0,   0, 255, 255, 255,   0,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [255,255, 0,   0,   0, 255,  255],
    [0,   0,   0,   0,   0,   0,   0]
], dtype=np.uint8)

img_big = cv2.resize(img, (400,400), interpolation=cv2.INTER_NEAREST)

ys, xs = np.where(img == 255)
M00 = len(xs)
M10 = np.sum(xs)
M01 = np.sum(ys)
M11 = np.sum(xs * ys)
M20 = np.sum(xs**2)
M02 = np.sum(ys**2)
x_centroid = M10 / M00
y_centroid = M01 / M00
u11 = M11 / M00 - x_centroid * y_centroid
u20 = M20 / M00 - x_centroid**2
u02 = M02 / M00 - y_centroid**2
theta = 0.5 * np.arctan2(2 * u11, u20 - u02)  # in radians
print("Kết quả lập trình tính tay")
print("M00: ", M00, " M10: ", M10, " M01: ", M01)
print("M11 = ", M11, " M20 = ", M20, " M02 = ", M02)
print("Area: ", M00, " pixels")
print("Centroid: ", (x_centroid, y_centroid), "≈", (int(x_centroid), int(y_centroid)))
print("Orientation angle (degrees): ", np.degrees(theta))

print("------------------------------")
print("Kết quả sử dụng OpenCV")

# Assuming img is your binary image (0 and 255)
contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # Centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"Centroid: ({cx}, {cy})")
    
    # Area
    area = cv2.contourArea(contour)
    print(f"Area: {area}")
    
    # Perimeter
    perimeter = cv2.arcLength(contour, True)
    print(f"Perimeter: {perimeter}")
    
    # Orientation 
    if len(contour) >= 5:  
        ellipse = cv2.fitEllipse(contour)
        angle = ellipse[2]
        print(f"Orientation angle: {angle} degrees")
