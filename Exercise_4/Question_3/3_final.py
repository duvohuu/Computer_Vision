import cv2
import numpy as np

# Step 1: Load video and image resources
def load_resources(video_path, image_path):
    source_image = cv2.imread(image_path)
    if source_image is None:
        raise ValueError("Không thể đọc hình ảnh nguồn!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video!")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    h, w = source_image.shape[:2]
    source_is_horizontal = w > h

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Source image: {w}x{h}, Orientation: {'Horizontal' if source_is_horizontal else 'Vertical'}")

    return cap, source_image, fps, width, height, total_frames, source_is_horizontal


# Step 2: Detect blue rectangle in each frame
def detect_blue_rectangle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 1000:
        return None

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)

    return approx.reshape(4, 2).astype("float32")


# Step 3: Order points of 4 corners (TL, TR, BR, BL - clockwise) 
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # TL
    rect[2] = pts[np.argmax(s)]     # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


# Step 4: Tracking – match with previous frame
def match_corners_with_previous(current, prev):
     """
     Thực hiện tính toán khoảng cách từ từng đỉnh của frame sau với từng đỉnh của frame trước, để tìm ra các đỉnh tương ứng, trách ảnh khi warping vào bị lật

     Args:
          current (ndarray): coors of 4 corners in current frame
          prev (ndarray): coors of 4 corners in previous frame

     Returns:
          ndarray: coors of 4 corners in current frame after arrange
     """
     matched = np.zeros_like(prev)
     used = [False] * 4

     for i in range(4):
          min_dist = float("inf")
          min_idx = -1
          for j in range(4):
               if used[j]:
                    continue
               dist = np.linalg.norm(prev[i] - current[j])
               if dist < min_dist:
                    min_dist = dist
                    min_idx = j

          matched[i] = current[min_idx]
          used[min_idx] = True

     return matched


# Step 5: Determine rectangle orientation
def get_rectangle_orientation(corners):
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[3] - corners[0])
    return width > height


# Step 6: Align source image corners with detected rectangle corners
def align_corners(src_corners, dst_corners, source_is_horizontal):
    dst_is_horizontal = get_rectangle_orientation(dst_corners)

    if source_is_horizontal == dst_is_horizontal:
        return src_corners.copy()

    # Rotate 90 degrees
    return np.roll(src_corners, 1, axis=0)


# Step 7: Warp source image into detected rectangle
def warp_image(frame, corners, source_image, source_is_horizontal):
    if corners is None:
        return frame

    h, w = source_image.shape[:2]
    src_points = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    aligned_src = align_corners(src_points, corners, source_is_horizontal)
    dst_points = corners.astype("float32")

    M = cv2.getPerspectiveTransform(aligned_src, dst_points)

    warped = cv2.warpPerspective(source_image, M, (frame.shape[1], frame.shape[0]))

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corners), 255)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask3))
    fg = cv2.bitwise_and(warped, mask3)

    return cv2.add(bg, fg)


# Step 8: Main processing loop
def process_video(video_path, image_path, output_path, show_preview=False):
    cap, source_image, fps, width, height, total_frames, source_is_horizontal = \
        load_resources(video_path, image_path)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    prev_corners = None
    is_initialized = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected = detect_blue_rectangle(frame)

        if detected is not None:
            if not is_initialized:
                prev_corners = order_points(detected)
                is_initialized = True
            else:
                prev_corners = match_corners_with_previous(detected, prev_corners)

        if prev_corners is not None:
            frame_output = warp_image(frame, prev_corners, source_image, source_is_horizontal)
        else:
            frame_output = frame

        out.write(frame_output)
        frame_count += 1

        if show_preview:
            debug = frame_output.copy()
            if prev_corners is not None:
                for i, c in enumerate(prev_corners):
                    cv2.circle(debug, tuple(c.astype(int)), 8, (0, 0, 255), -1)
            cv2.imshow("Preview", debug)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Hoàn thành! Video lưu tại: {output_path}")


# Main
if __name__ == "__main__":
    VIDEO_PATH = r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\Presentation1.mp4"
    IMAGE_PATH = r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\Ex4_Q3.png"
    OUTPUT_PATH = r"output_video.mp4"

    process_video(VIDEO_PATH, IMAGE_PATH, OUTPUT_PATH, show_preview=True)
