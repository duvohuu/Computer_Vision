import cv2
import numpy as np

class ImageWarper:
    def __init__(self, video_path, image_path, output_path):
        """
        Khởi tạo Image Warper
        
        Args:
            video_path: Đường dẫn video input
            image_path: Đường dẫn hình ảnh cần chiếu
            output_path: Đường dẫn video output
        """
        self.video_path = video_path
        self.image_path = image_path
        self.output_path = output_path
        
        # Đọc hình ảnh nguồn
        self.source_image = cv2.imread(image_path)
        if self.source_image is None:
            raise ValueError("Không thể đọc hình ảnh nguồn!")
        
        # Mở video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Không thể mở video!")
        
        # Lấy thông số video
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames")
        
    def detect_blue_rectangle(self, frame):
        """
        Phát hiện vùng màu xanh dương trong frame
        
        Returns:
            corners: 4 góc của hình chữ nhật (numpy array)
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Định nghĩa range màu xanh dương
        # Điều chỉnh giá trị này tùy theo màu xanh cụ thể
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Tạo mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphological operations để làm sạch mask
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
        if area < 1000:  # Threshold có thể điều chỉnh
            return None
        
        # Approximate thành polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Nếu không phải 4 góc, dùng minAreaRect
        if len(approx) != 4:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)
        
        # Sắp xếp các góc theo thứ tự: top-left, top-right, bottom-right, bottom-left
        corners = self.order_points(approx.reshape(4, 2))
        
        return corners
    
    def order_points(self, pts):
        """
        Sắp xếp 4 điểm theo thứ tự: TL, TR, BR, BL
        """
        # Sắp xếp theo tổng x+y
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left có tổng nhỏ nhất
        rect[2] = pts[np.argmax(s)]  # Bottom-right có tổng lớn nhất
        
        # Sắp xếp theo hiệu x-y
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect
    
    def warp_image(self, frame, corners):
        """
        Warp hình ảnh vào vùng được xác định bởi corners
        
        Args:
            frame: Frame gốc
            corners: 4 góc của vùng đích
            
        Returns:
            frame_with_warp: Frame đã được warp
        """
        if corners is None:
            return frame
        
        # Định nghĩa 4 góc của hình ảnh nguồn
        h, w = self.source_image.shape[:2]
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
        warped = cv2.warpPerspective(
            self.source_image, 
            M, 
            (frame.shape[1], frame.shape[0])
        )
        
        # Tạo mask cho vùng warped
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(corners), 255)
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Composite: Xóa vùng cũ và thêm hình mới
        frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask_3channel))
        frame_fg = cv2.bitwise_and(warped, mask_3channel)
        result = cv2.add(frame_bg, frame_fg)
        
        return result
    
    def process_video(self, show_preview=False):
        """
        Xử lý toàn bộ video
        
        Args:
            show_preview: Hiển thị preview trong khi xử lý (chậm hơn)
        """
        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        frame_count = 0
        prev_corners = None  # Lưu corners của frame trước để xử lý khi detect fail
        
        print("Bắt đầu xử lý video...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect vùng xanh
            corners = self.detect_blue_rectangle(frame)
            
            # Nếu không detect được, dùng corners của frame trước
            if corners is None and prev_corners is not None:
                corners = prev_corners
            
            # Warp hình ảnh
            if corners is not None:
                frame_output = self.warp_image(frame, corners)
                prev_corners = corners
            else:
                frame_output = frame
            
            # Ghi frame
            out.write(frame_output)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Đã xử lý {frame_count}/{self.total_frames} frames")
            
            # Hiển thị preview (optional)
            if show_preview:
                cv2.imshow('Preview', frame_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Giải phóng resources
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Hoàn thành! Video đã được lưu tại: {self.output_path}")


# ========== CÁCH SỬ DỤNG ==========
if __name__ == "__main__":
    # Đường dẫn files
    VIDEO_PATH = r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\Presentation1.mp4"
    IMAGE_PATH = r"D:\Fundemental\4\251\Computer_Vision\Computer_Vision\Exercise_4\Question_3\Ex4_Q3.png"
    OUTPUT_PATH = r"output_video.mp4"
    
    try:
        # Khởi tạo warper
        warper = ImageWarper(VIDEO_PATH, IMAGE_PATH, OUTPUT_PATH)
        
        # Xử lý video (set show_preview=True để xem preview)
        warper.process_video(show_preview=True)
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")