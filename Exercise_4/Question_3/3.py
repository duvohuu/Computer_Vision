import cv2
import numpy as np


def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự: TL, TR, BR, BL
    
    Args:
        pts: Array 4 điểm
        
    Returns:
        rect: Array 4 điểm đã sắp xếp
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left có tổng nhỏ nhất
    rect[2] = pts[np.argmax(s)]  # Bottom-right có tổng lớn nhất
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect


def detect_blue_rectangle(frame):
    """
    Phát hiện vùng màu xanh dương trong frame (xử lý trên ảnh xám)
    
    Args:
        frame: Frame màu từ video
        
    Returns:
        corners: 4 góc của hình chữ nhật hoặc None
    """
    # Chuyển sang HSV để detect màu
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Range màu xanh dương (điều chỉnh theo màu cụ thể)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Tạo mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Morphological operations làm sạch
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Kiểm tra diện tích tối thiểu
    area = cv2.contourArea(largest_contour)
    if area < 1000:
        return None
    
    # Approximate thành polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Nếu không phải 4 góc, dùng minAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)
    
    # Sắp xếp các góc
    corners = order_points(approx.reshape(4, 2))
    
    return corners


def warp_image_to_frame(frame, source_image, corners):
    """
    Warp hình ảnh vào frame tại vị trí corners
    
    Args:
        frame: Frame gốc (có thể là màu hoặc xám)
        source_image: Hình ảnh cần warp
        corners: 4 góc vùng đích
        
    Returns:
        result: Frame đã được warp
    """
    if corners is None:
        return frame
    
    # Kiểm tra frame có phải ảnh xám không
    is_gray = len(frame.shape) == 2
    
    # Chuyển source image sang xám nếu frame là xám
    if is_gray and len(source_image.shape) == 3:
        source_image_processed = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    else:
        source_image_processed = source_image.copy()
    
    # Định nghĩa 4 góc của hình ảnh nguồn
    h, w = source_image_processed.shape[:2]
    src_points = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")
    
    # Tính ma trận homography
    dst_points = corners.astype("float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Warp hình ảnh
    if is_gray:
        warped = cv2.warpPerspective(
            source_image_processed, 
            M, 
            (frame.shape[1], frame.shape[0])
        )
    else:
        warped = cv2.warpPerspective(
            source_image_processed, 
            M, 
            (frame.shape[1], frame.shape[0])
        )
    
    # Tạo mask cho vùng warped
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corners), 255)
    
    # Composite
    if is_gray:
        frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        frame_fg = cv2.bitwise_and(warped, mask)
    else:
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask_3channel))
        frame_fg = cv2.bitwise_and(warped, mask_3channel)
    
    result = cv2.add(frame_bg, frame_fg)
    
    return result


def process_video(video_path, image_path, output_path, use_grayscale=True, show_preview=False):
    """
    Xử lý video: warp hình ảnh vào vùng màu xanh
    
    Args:
        video_path: Đường dẫn video input
        image_path: Đường dẫn hình ảnh nguồn
        output_path: Đường dẫn video output
        use_grayscale: True để xử lý trên ảnh xám
        show_preview: True để hiển thị preview
        
    Returns:
        success: True nếu xử lý thành công
    """
    # Đọc hình ảnh nguồn
    source_image = cv2.imread(image_path)
    if source_image is None:
        print("Lỗi: Không thể đọc hình ảnh nguồn!")
        return False
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video!")
        return False
    
    # Lấy thông số video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Chế độ: {'Ảnh xám' if use_grayscale else 'Màu'}")
    
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), 
                          isColor=not use_grayscale)
    
    frame_count = 0
    prev_corners = None
    
    print("Bắt đầu xử lý video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vùng xanh (luôn trên frame màu)
        corners = detect_blue_rectangle(frame)
        
        # Nếu không detect được, dùng corners trước
        if corners is None and prev_corners is not None:
            corners = prev_corners
        
        # Chuyển frame sang xám nếu cần
        if use_grayscale:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_processed = frame.copy()
        
        # Warp hình ảnh
        if corners is not None:
            frame_output = warp_image_to_frame(frame_processed, source_image, corners)
            prev_corners = corners
        else:
            frame_output = frame_processed
        
        # Ghi frame
        out.write(frame_output)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count}/{total_frames} frames")
        
        # Hiển thị preview
        if show_preview:
            cv2.imshow('Preview', frame_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Dừng xử lý bởi người dùng")
                break
    
    # Giải phóng resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Hoàn thành! Video đã được lưu tại: {output_path}")
    return True


# ========== CÁCH SỬ DỤNG ==========
if __name__ == "__main__":
    # Đường dẫn files
    VIDEO_PATH = r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\Presentation1.mp4"
    IMAGE_PATH = r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\Ex4_Q3.png"
    OUTPUT_PATH = r"output_video.mp4"
    
    # Xử lý video với ảnh xám
    success = process_video(
        video_path=VIDEO_PATH,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        use_grayscale=True,      # True = xám, False = màu
        show_preview=False       # True để xem preview
    )
    
    if success:
        print("Xử lý thành công!")
    else:
        print("Xử lý thất bại!")