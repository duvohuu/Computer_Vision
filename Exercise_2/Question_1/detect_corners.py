import cv2
import numpy as np


'''
Function : cv2.cornerHarris(image,blocksize,ksize,k)
Parameters are as follows :
1. image : the source image in which we wish to find the corners (grayscale)
2. blocksize : size of the neighborhood in which we compare the gradient
3. ksize : aperture parameter for the Sobel() Operator (used for finding Ix & Iy)
4. k : Harris detector free parameter (used in computing R)
'''

def harris_corners(image, blockSize, ksize, k, threshold):

    if len(image.shape) == 3: 
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # sigma = 0.5 * 5
    # gray = cv2.GaussianBlur(gray_img, (3, 3), sigma, borderType=cv2.BORDER_REPLICATE)
    gray = np.float32(gray_img)

    # Harris Corner Detection
    R_raw = cv2.cornerHarris(gray, blockSize, ksize, k, borderType=cv2.BORDER_REPLICATE)
    
    if image.ndim == 2:
        R_norm = cv2.normalize(cv2.normalize(R_raw, None, 0, 255, cv2.NORM_MINMAX), None, 0, 255, cv2.NORM_MINMAX)
        R_scaled = cv2.convertScaleAbs(R_norm)

        _, R_thresh = cv2.threshold(R_scaled, threshold, 255, cv2.THRESH_BINARY)

        return R_norm, R_thresh

    return image.copy(), R_raw


'''
Function: cv2.goodFeaturesToTrack(image,maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
image – Input 8-bit or floating-point 32-bit (grayscale image).
maxCorners – You can specify the maximum no. of corners to be detected. (Strongest ones are returned if detected more than max.)
qualityLevel – Minimum accepted quality of image corners.
minDistance – Minimum possible Euclidean distance between the returned corners.
corners – Output vector of detected corners.
mask – Optional region of interest. 
blockSize – Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. 
useHarrisDetector – Set this to True if you want to use Harris Detector with this function.
k – Free parameter of the Harris detector (used in computing R)
'''


def shi_tomasi(image, maxCornersNB=2, qualityLevel=0.01, minDistance=1, blockSize=3):
    """
    Detect corners using Shi-Tomasi (goodFeaturesToTrack)
    Returns:
        corners_coords: list of (x, y)
        result_binary: binary matrix with corner points = 255
    """

    # Chuyển ảnh sang grayscale nếu cần
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    gray_img = np.float32(gray_img)

    # Gọi goodFeaturesToTrack (Shi–Tomasi)
    corners = cv2.goodFeaturesToTrack(
        gray_img,
        maxCorners=maxCornersNB,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
        mask=None,
        blockSize=blockSize,
        useHarrisDetector=False
    )

    # Khởi tạo ma trận kết quả nhị phân
    result_binary = np.zeros_like(gray_img, dtype=np.uint8)

    # Nếu phát hiện được góc → đánh dấu bằng 255
    corners_coords = []
    if corners is not None:
        corners = corners.astype(int)
        for c in corners:
            x, y = c.ravel()
            if 0 <= y < result_binary.shape[0] and 0 <= x < result_binary.shape[1]:
                result_binary[y, x] = 255
                corners_coords.append((x, y))

    # Trả về danh sách góc + ma trận nhị phân
    return corners_coords, result_binary
