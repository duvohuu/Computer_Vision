import cv2
import numpy as np

# ================== CONFIG ==================
IMAGE_PATH = "E:/251/Computer_Vision/Repo/Computer_Vision/Exercise_3/Question_3/parking.png"
MAX_SIDE_PX  = 900      # resize cho nhẹ
SHOW_DEBUG   = True     # False nếu không muốn hiện từng cửa sổ
CROP_TOP_PX  = 0        # nếu ảnh có chữ phía trên thì cắt bớt

# ===== THAM SỐ DETECT VẠCH TRẮNG / HOUGH =====
L_MIN        = 165      # L trong HLS >= L_MIN
S_MAX        = 160      # S trong HLS <= S_MAX
GRAY_BRIGHT  = 170      # gray >= GRAY_BRIGHT để coi là sáng

HOUGH_THRESH = 30       # ngưỡng HoughLinesP (số điểm trên 1 line)
HOUGH_MINLEN = 50       # minLineLength (px)  -- GIẢM để bắt vạch ngắn
HOUGH_MAXGAP = 30       # maxLineGap (px)    -- TĂNG để nối vạch đứt

ANGLE_HORIZ  = 30       # |góc| < ANGLE_HORIZ -> ngang
ANGLE_VERT_L = 60       # ANGLE_VERT_L < |góc| < ANGLE_VERT_H -> dọc
ANGLE_VERT_H = 120

MIN_SLOT_W   = 20       # lọc ô nhỏ quá
MIN_SLOT_H   = 20


# ================== TIỀN XỬ LÝ ==================
def gaussian_canny(img, blur_ksize=(5, 5), t1=80, t2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    edges = cv2.Canny(blur, t1, t2)
    return edges


def estimate_main_angle(edges):
    """
    HoughLines trên edge để tìm các line chéo lớn (thân bãi),
    trả về góc trung bình (degree).
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    angles = []
    if lines is not None:
        for l in lines:
            rho, theta = l[0]
            ang = np.degrees(theta)
            # giữ line chéo (không gần 0 hoặc 90)
            if 15 < ang < 75 or 105 < ang < 165:
                angles.append(ang)
    if angles:
        return float(np.median(angles))
    else:
        return 90.0   # không tìm được thì coi như đứng


def rotate_image(img, angle_deg):
    """
    Xoay ảnh quanh tâm. angle_deg dương: CCW.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated, M


# ================== GRID Ô NHỎ (TỪ VẠCH TRẮNG) ==================
def detect_grid_slots_from_img(rot_img):
    """
    1. Tạo mask vạch trắng bằng HLS + gray.
    2. Canny + HoughLinesP -> line ngang/dọc -> xs, ys.
    3. Refine vị trí vạch dọc bằng mask_HLS ở vùng đáy ảnh.
    4. Nếu chỉ có 2 vạch dọc thì sinh thêm 1 vạch bên trái.
    5. Dùng xs, ys để build list các ô nhỏ (slots).
    """
    # --- HLS ---
    hls = cv2.cvtColor(rot_img, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(hls)

    mask_L   = cv2.inRange(L, L_MIN, 255)
    mask_S   = cv2.inRange(S, 0, S_MAX)
    mask_hls = cv2.bitwise_and(mask_L, mask_S)  # vùng trắng/xám sáng

    # --- Gray + Canny để cứu những vạch mờ ---
    gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
    edges_gray   = cv2.Canny(gray, 70, 180)
    mask_bright  = cv2.inRange(gray, GRAY_BRIGHT, 255)
    mask_edge_bright = cv2.bitwise_and(edges_gray, mask_bright)

    # --- union 2 mask ---
    mask_white = cv2.bitwise_or(mask_hls, mask_edge_bright)

    # dọn noise, nối vạch mỏng
    kernel = np.ones((3, 3), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, 1)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, 1)

    # --- Canny trên mask này ---
    edges = cv2.Canny(mask_white, 40, 120)

    # --- HoughLinesP ---
    linesP = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        HOUGH_THRESH,
        minLineLength=HOUGH_MINLEN,
        maxLineGap=HOUGH_MAXGAP
    )

    vertical_x   = []
    horizontal_y = []

    if linesP is not None:
        for l in linesP:
            x1, y1, x2, y2 = l[0]
            dx, dy = x2 - x1, y2 - y1
            ang = np.degrees(np.arctan2(dy, dx))

            if abs(ang) < ANGLE_HORIZ:  # gần ngang
                horizontal_y.append((y1 + y2) / 2.0)
            elif ANGLE_VERT_L < abs(ang) < ANGLE_VERT_H:  # gần dọc
                vertical_x.append((x1 + x2) / 2.0)

    def merge_coords(coords, thresh):
        if not coords:
            return []
        coords = sorted(coords)
        merged = [coords[0]]
        for c in coords[1:]:
            if abs(c - merged[-1]) > thresh:
                merged.append(c)
            else:
                merged[-1] = (merged[-1] + c) / 2.0
        return merged

    # gộp vạch dọc/ngang
    xs_rough = merge_coords(vertical_x, 60)
    ys       = merge_coords(horizontal_y, 20)

    # ======= REFINE VỊ TRÍ VẠCH DỌC BẰNG TỔNG PIXEL TRẮNG THEO CỘT =======
    h, w = mask_hls.shape
    refined_xs = []
    search_win = 70
    y0 = int(h * 0.6)      # vùng đáy ảnh

    for xr in xs_rough:
        x0 = max(int(xr) - search_win, 0)
        x1 = min(int(xr) + search_win, w - 1)

        roi_bottom = mask_hls[y0:h, x0:x1]
        col_sum = roi_bottom.sum(axis=0)

        if col_sum.max() > 0:
            best_offset = int(np.argmax(col_sum))
            x_ref = x0 + best_offset
        else:
            # fallback: dùng toàn chiều cao
            roi_all = mask_hls[:, x0:x1]
            col_sum_all = roi_all.sum(axis=0)
            if col_sum_all.max() > 0:
                best_offset = int(np.argmax(col_sum_all))
                x_ref = x0 + best_offset
            else:
                x_ref = float(xr)

        refined_xs.append(float(x_ref))

    xs = sorted(refined_xs)

    # --------- PHƯƠNG ÁN 1: sinh thêm 1 vạch dọc nếu chỉ có 2 vạch ---------
    if len(xs) == 2:
        col_w = xs[1] - xs[0]
        if col_w > MIN_SLOT_W:
            x_left = xs[0] - col_w
            if x_left > 0:
                xs.insert(0, x_left)

    print("vertical_x refined xs:", xs)
    print("horizontal_y merged ys:", ys)

    # ======= BUILD Ô NHỎ =======
    slots = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1 = int(xs[i])
            x2 = int(xs[i + 1])
            y1 = int(ys[j])
            y2 = int(ys[j + 1])
            if x2 - x1 < MIN_SLOT_W or y2 - y1 < MIN_SLOT_H:
                continue
            slots.append((x1, y1, x2, y2))

    return slots, xs, ys, mask_white


# ================== PHÂN LOẠI SLOT TRỐNG / CÓ XE ==================
def classify_slots(rot_img, slots,
                   s_thresh=50,
                   area_ratio_thresh=0.12):
    """
    Dùng kênh S (saturation) trong HSV:
    - Xe nhiều màu -> nhiều pixel S > s_thresh
    - Slot trống -> S thấp
    """
    occupied = []
    empty    = []

    for (x1, y1, x2, y2) in slots:
        slot = rot_img[y1:y2, x1:x2]
        if slot.size == 0:
            continue

        hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
        _, S, _ = cv2.split(hsv)

        mask = (S > s_thresh).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

        ratio = mask.mean()  # 0..1

        if ratio > area_ratio_thresh:
            occupied.append((x1, y1, x2, y2, ratio))
        else:
            empty.append((x1, y1, x2, y2, ratio))

    return occupied, empty


# ================== VẼ KẾT QUẢ TRÊN ẢNH XOAY ==================
def draw_result(rot_img, occupied, empty):
    out = rot_img.copy()

    for (x1, y1, x2, y2, _) in occupied:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # đỏ

    for (x1, y1, x2, y2, _) in empty:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)  # xanh

    return out


# ================== MAP SLOT NGƯỢC VỀ ẢNH GỐC ==================
def transform_points_quad(x1, y1, x2, y2, M_inv):
    """
    Biến đổi 4 góc của ô (x1,y1,x2,y2) từ hệ ảnh xoay -> ảnh gốc.
    Trả về array (4,2).
    """
    pts_rot = np.array([[
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]], dtype=np.float32)  # (1,4,2)

    pts_orig = cv2.transform(pts_rot, M_inv)
    pts_orig = np.round(pts_orig).astype(np.int32)
    return pts_orig[0]  # (4,2)


def draw_on_original(orig_img, occupied, empty, M_inv):
    out = orig_img.copy()

    for (x1, y1, x2, y2, _) in empty:
        quad = transform_points_quad(x1, y1, x2, y2, M_inv)
        cv2.polylines(out, [quad.reshape((-1, 1, 2))],
                      isClosed=True, color=(0, 255, 0), thickness=2)

    for (x1, y1, x2, y2, _) in occupied:
        quad = transform_points_quad(x1, y1, x2, y2, M_inv)
        cv2.polylines(out, [quad.reshape((-1, 1, 2))],
                      isClosed=True, color=(0, 0, 255), thickness=2)

    return out


# ================== MAIN ==================
def main():
    img0 = cv2.imread(IMAGE_PATH)
    if img0 is None:
        print("Không đọc được ảnh, kiểm tra lại IMAGE_PATH")
        return

    if CROP_TOP_PX > 0:
        img0 = img0[CROP_TOP_PX:, :]

    # resize
    h0, w0 = img0.shape[:2]
    max_side = max(h0, w0)
    if max_side > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / max_side
        img = cv2.resize(img0, (int(w0 * scale), int(h0 * scale)))
    else:
        img = img0.copy()

    # 1. Canny để estimate góc nghiêng bãi
    edges0 = gaussian_canny(img)
    main_angle = estimate_main_angle(edges0)
    delta_angle = main_angle - 90.0

    # 2. Xoay ảnh
    rot, M = rotate_image(img, delta_angle)

    # 3. Grid ô nhỏ từ vạch trắng
    slots, xs, ys, mask_white = detect_grid_slots_from_img(rot)

    # 4. Phân loại slot trống / có xe
    occupied, empty = classify_slots(rot, slots)

    print("Góc chính (deg):", main_angle)
    print("Delta xoay:", delta_angle)
    print("Số slot:", len(slots))
    print("Slot có xe:", len(occupied))
    print("Slot trống:", len(empty))

    # 5. Vẽ trên ảnh xoay
    result_rot = draw_result(rot, occupied, empty)

    # 6. Vẽ slot về lại ảnh gốc
    M_inv = cv2.invertAffineTransform(M)
    result_orig = draw_on_original(img, occupied, empty, M_inv)

    if SHOW_DEBUG:
        cv2.imshow("Original (after crop/resize)", img)
        cv2.imshow("Edges original", edges0)
        cv2.imshow("Rotated", rot)
        cv2.imshow("White mask (rotated)", mask_white)
        cv2.imshow("Result on rotated", result_rot)
        cv2.imshow("Result on original orientation", result_orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite("Computer_Vision/Exercise_3/Question_3/parking_result_rotateddgdgdf.png", result_rot)
    cv2.imwrite("Computer_Vision/Exercise_3/Question_3/parking_result_original.png", result_orig)
    print("Đã lưu: parking_result_rotated.png, parking_result_original.png")


if __name__ == "__main__":
    main()
