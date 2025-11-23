import cv2
import numpy as np

def create_calibration_data(image_path):
    """
    GUI creates calibration data by allowing the user to click on points in the image.
    Returns a list of pixel coordinates.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot load image.")
        return None

    # List to store pixel coordinates
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Clicked point: ({x}, {y})")
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', img)

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)

    print("Click points on the image. Press 'q' to stop.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if not points:
        print("No points were clicked.")
        return None

    return points

# Read calibration data from images
if __name__ == "__main__":
    image_path_h1 = r"D:\Fundemental\4\251\Computer_Vision\BTL\Bai5\pictures\492.jpg"  # Image path
    image_path_h2 = r"D:\Fundemental\4\251\Computer_Vision\BTL\Bai5\pictures\475.jpg"
    image_path_h3 = r"D:\Fundemental\4\251\Computer_Vision\BTL\Bai5\pictures\467.jpg"

    data_h1 = create_calibration_data(image_path_h1)
    data_h2 = create_calibration_data(image_path_h2)
    data_h3 = create_calibration_data(image_path_h3)
    
    # World coordinates corresponding to the images coords
    data = np.array([[80, 30, 492],
                    [50, 50, 492],
                    [-40, 25, 492],
                    [-60, -20, 492],
                    [50, -40, 492],
                    [80, 30, 475],
                    [50, 50, 475],
                    [-40, 25, 475],
                    [-60, -20, 475],
                    [50, -40, 475],
                    [80, 30, 467],
                    [50, 50, 467],
                    [-40, 25, 467],
                    [-60, -20, 467],
                    [50, -40, 467]])
    
    data = np.hstack((np.array(data_h1 + data_h2 + data_h3), data))
    
    print("Data from image 1: h1 = 492 mm:", data_h1)
    print("Data from image 2: h2 = 475 mm:", data_h2)
    print("Data from image 3: h3 = 467 mm:", data_h3)

    np.savetxt("calibration_data.txt", data, fmt="%.6f")
    if data is not None:
        print("Calibration data:")
        print(data)
        
    
        
        
# Get image coordinates and world coordinates
imgCoor = data[:, :2]  # columns u_1, v_1
worldCoor = data[:, 2:]  # columns x_1, y_1, z_1

# Build matrix P
P = np.zeros((2*len(imgCoor), 12))
j = 0

for i in range(0, 2*len(imgCoor), 2):
    P[i, 0:3] = worldCoor[j, 0:3]
    P[i+1, 4:7] = worldCoor[j, 0:3]
    P[i, 3] = 1
    P[i+1, 7] = 1
    P[i, 8:12] = -np.concatenate([worldCoor[j, 0:3], [1]]) * imgCoor[j, 0]
    P[i+1, 8:12] = -np.concatenate([worldCoor[j, 0:3], [1]]) * imgCoor[j, 1]
    j += 1

# SVD decomposition
U, S, V = np.linalg.svd(P)
V_last = V[-1, :]

# Create matrix M
M = np.zeros((3, 4))
M[0, 0:4] = V_last[0:4]
M[1, 0:4] = V_last[4:8]
M[2, 0:4] = V_last[8:12]

print('Projected matrix')
M_ext = M.copy()
print(M_ext)

# Extract intrinsic and extrinsic matrices from matrix M
M = M / M[2, 3]
H = M[:, 0:3]  # first 3 columns of M
T = M[:, 3]    # translation matrix

# QR decomposition to find K and Ro
Q, R = np.linalg.qr(np.linalg.inv(H))
K = np.linalg.inv(R)
Ro = Q.T

# K must have positive diagonal elements
t = np.array([[-1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
K = K @ t / K[2, 2]

print('\nIntrinsic matrix K:')
print(K)

print('\nExtrinsics Matrix:')
EXT = np.column_stack([Ro, T])
print(EXT)






# Calculate world coordinates from image coordinates
z_known = 492  # distance from camera to surface

image_path_test = r"D:\Fundemental\4\251\Computer_Vision\BTL\Bai5\pictures\492.jpg"

worldCoor_3D = []
# imgCoor_test = np.array([[845, 507], [1462, 100], [1750, 412]])
imgCoor_test = create_calibration_data(image_path_test)
imgCoor_test = np.array(imgCoor_test)
point_names = ['P1', 'P2', 'P3']

print('\n--------------------------------------------')
for i in range(len(imgCoor_test)):
    u = imgCoor_test[i, 0]
    v = imgCoor_test[i, 1]
    
    # Define the system of equations
    A = np.array([
        [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1]],
        [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1]]
    ])
    
    B = np.array([
        u*M[2,2]*z_known + u*M[2,3] - M[0,2]*z_known - M[0,3],
        v*M[2,2]*z_known + v*M[2,3] - M[1,2]*z_known - M[1,3]
    ])
    
    # Solve for world coordinates (X, Y)
    world_pt = np.linalg.solve(A, B)
    worldCoor_3D.append([np.ceil(world_pt[0]), np.ceil(world_pt[1]), z_known])
    
    print(f'Img_coordinate at {point_names[i]}: ({int(u)}, {int(v)}) --------> '
          f'World_Coordinate: ({int(np.ceil(world_pt[0]))}, {int(np.ceil(world_pt[1]))})')

worldCoor_3D = np.array(worldCoor_3D)

# Calculate Euclidean distance between P1 and P2
p1 = worldCoor_3D[0]
p2 = worldCoor_3D[1]

distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

print('--------------------------------------------')
print(f'Distance between P1({int(p1[0])}, {int(p1[1])}) and P2({int(p2[0])}, {int(p2[1])}) '
      f'with world_Coordinate: {distance:.6f} mm')







