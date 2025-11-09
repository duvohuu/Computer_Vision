import numpy as np
import cv2
from detect_corners import harris_corners , shi_tomasi

def find_corners(img):
    # --- Harris Corner ---
    R_norm, R_thresh = harris_corners(img, blockSize=3, ksize=3, k=0.06, threshold=250)
    print("Ma trận Harris sau chuẩn hóa:")
    print(R_norm)
    print("Ngưỡng Harris:")
    print(R_thresh)

    # --- Shi-Tomasi Corner ---
    corners, shi_thresh = shi_tomasi(img, maxCornersNB=2, qualityLevel=0.01, minDistance=1)
    print("\nVị trí góc Shi-Tomasi:", corners)
    print("Ma trận nhị phân Shi-Tomasi:")
    print(shi_thresh)

def main():
    matrix = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 0, 0, 0],
        [0, 0, 255, 255, 255, 0, 0],
        [0, 255, 255, 255, 255, 255, 0],
        [0, 255, 255, 255, 255, 255, 0],
        [255, 255, 0, 0, 0, 255, 255],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    find_corners(matrix)

if __name__ == "__main__":
    main()  
