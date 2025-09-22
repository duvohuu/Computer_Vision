import cv2
import numpy as np
import math

# Load image
notArmImage = cv2.imread("pictures/NotArm.png")
armImage = cv2.imread("pictures/Arm.png")

# Check if images are loaded successfully
if notArmImage is None:
    print("Error: Could not load background image (NotArm.png)")
    exit(-1)

if armImage is None:
    print("Error: Could not load current image (Arm.png)")
    exit(-1)

# Resize if dimensions of 2 images are different
if notArmImage.shape[:2] != armImage.shape[:2]:
    notArmImage = cv2.resize(notArmImage, (armImage.shape[1], armImage.shape[0]))

# Calculate diffImage
diffImage = cv2.absdiff(notArmImage, armImage)


# Image to show only the hand (color)
handImage = np.zeros_like(armImage)

threshold = 30
distMat = np.zeros((diffImage.shape[0], diffImage.shape[1]), dtype=np.float64)

# Calculate Euclidean distance for each pixel
for j in range(diffImage.shape[0]):
    for i in range(diffImage.shape[1]):
        pix = diffImage[j, i]
        dist = math.sqrt(int(pix[0])**2 + int(pix[1])**2 + int(pix[2])**2)
        distMat[j, i] = dist
        if dist > threshold:
            handImage[j, i] = armImage[j, i]   

# Print distance matrix
np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
print("Distance Matrix:")
print(distMat)

# Normalize distance matrix to display as grayscale image
distImage8U = cv2.normalize(distMat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# Show all the images
cv2.imshow("Background", notArmImage)
cv2.imshow("Current", armImage)
cv2.imshow("Diff", diffImage)
cv2.imshow("Hand (Color Extracted)", handImage)
cv2.imshow("Distance Matrix (Gray)", distImage8U)

cv2.waitKey(0)
cv2.destroyAllWindows()
