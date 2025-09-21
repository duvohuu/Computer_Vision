import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rgb_to_hsv_manual(r, g, b):
    """Tính HSV theo công thức chuẩn"""
    # Chuẩn hóa về [0,1]
    r, g, b = r/255.0, g/255.0, b/255.0
    
    # Tính V (Value)
    V = max(r, g, b)
    min_val = min(r, g, b)
    
    # Tính S (Saturation)
    if V != 0:
        S = (V - min_val) / V
    else:
        S = 0
    
    # Tính H (Hue)
    if V == min_val:  # Grayscale
        H = 0
    else:
        if V == r:  # Red is max
            H = 60 * (g - b) / (V - min_val)
        elif V == g:  # Green is max
            H = 120 + 60 * (b - r) / (V - min_val)
        else:  # Blue is max
            H = 240 + 60 * (r - g) / (V - min_val)
    
    # Đảm bảo H trong khoảng [0, 360)
    if H < 0:
        H += 360
    
    return H, S * 100, V * 100  # Trả về H(0-360), S(0-100), V(0-100)

def classify_lemon(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh {img_path}")
        return None, None, None

    # Tạo mask để loại bỏ background
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
    
    # Tính Hue theo công thức manual chuẩn
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_rgb = mask > 0
    
    if np.sum(mask_rgb) > 0:
        mean_b = np.mean(img[mask_rgb][:, 0])  # Blue channel
        mean_g = np.mean(img[mask_rgb][:, 1])  # Green channel  
        mean_r = np.mean(img[mask_rgb][:, 2])  # Red channel
        
        H, S, V = rgb_to_hsv_manual(mean_r, mean_g, mean_b)
        
        print(f"\n{img_path}")
        print(f"Mean RGB: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")
        print(f"Manual Hue: {H:.2f}°")
    else:
        H = 0
        print(f"No valid pixels found in mask for {img_path}")

    # Phân loại dựa trên Hue manual
    if 40 <= H <= 55:
        result = "Good (Yellow)"
    elif 56 < H <= 65:
        result = "Acceptable (Yellow-Green)"
    elif H > 65:
        result = "Bad (Green)"
    else:
        result = "Unknown"

    return img_rgb, result, H

# Danh sách ảnh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images = [
    (os.path.join(BASE_DIR, "lemonA.png"), "Lemon A"),
    (os.path.join(BASE_DIR, "lemonB.png"), "Lemon B"),
    (os.path.join(BASE_DIR, "lemonC.png"), "Lemon C")
]

plt.figure(figsize=(12, 4))

for i, (path, name) in enumerate(images, 1):
    img, result, H = classify_lemon(path)
    if img is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.title(f"{name}\n{result}\nHue={H:.2f}°")
        plt.axis("off")

plt.tight_layout()
plt.show()
