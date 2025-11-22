import cv2
import numpy as np

img = np.array([
    [0,   0,   0,   0,   0,   0,   0],
    [0,   0,   0, 255,   0,   0,   0],
    [0,   0, 255, 255, 255,   0,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [255,255, 0,   0,   0, 255,255],
    [0,   0,   0,   0,   0,   0,   0]
], dtype=np.uint8)

img_big = cv2.resize(img, (400,400), interpolation=cv2.INTER_NEAREST)
# cv2.imshow("Image", img_big)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


print(img)
ys, xs = np.where(img == 255)
print("ys = ", ys)
print("xs = ", xs)
M00 = len(xs)
M10 = np.sum(xs)
M01 = np.sum(ys)
x_centroid = M10 / M00
y_centroid = M01 / M00
print("M00: ", M00, " M10: ", M10, " M01: ", M01)
print("Centroid: ", (x_centroid, y_centroid))