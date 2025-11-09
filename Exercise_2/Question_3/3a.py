# star_morph_only_with_horizontal_v2.py
# Morphology-only: cạnh CHÉO + NGANG (gating 2 đầu) + LOẠI nét bên trong bằng CC lớn nhất.

import cv2 as cv
import numpy as np
import os

P = dict(
    # Binarize
    blur_k=5,

    # CHÉO: line-opening
    angle_step = 3,
    len_ratio = 0.18,        # chiều dài SE cho cạnh chéo
    thick = 3,               # bề dày kernel & khi vẽ lại
    diag_min_deg = 15,       # dải góc chéo
    diag_max_deg = 75,

    # Closing nhỏ cho chéo
    close_gap = 3,

    # NGANG: opening dài + lọc hình dạng + gating 2 đầu
    horiz_len_ratio = 0.25,
    min_horiz_len_ratio = 0.22,
    max_horiz_thick_ratio = 0.03,
    anchor_dilate_ratio = 0.02,

    # CC filtering
    keep_only_largest_cc = True,  # << bật lên để bỏ nét bên trong
    min_area_ratio = 0.12,        # nếu muốn giữ >1 CC, dùng tỉ lệ theo CC lớn nhất
)

# ---------- helper functions ----------

# Tạo SE hình đĩa bán kính r
def diskSE(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))

# Tạo kernel line dài L, dày t, góc ang_deg (độ)
def line_kernel(L: int, t: int, ang_deg: float) -> np.ndarray:
    L = max(5, int(L)); t = max(1, int(t))
    k = L + 2*t + 2
    ker = np.zeros((k, k), np.uint8)
    c = k // 2
    rad = np.deg2rad(ang_deg)
    dx, dy = np.cos(rad)*(L/2), np.sin(rad)*(L/2)
    x1, y1 = int(round(c - dx)), int(round(c - dy))
    x2, y2 = int(round(c + dx)), int(round(c + dy))
    cv.line(ker, (x1, y1), (x2, y2), 255, thickness=t)
    return (ker > 0).astype(np.uint8)

# Kiểm tra góc a có nằm trong dải [amin, amax] (độ) không, xét cả 2 hướng
def in_diag(a: float, amin: float, amax: float) -> bool:
    a = abs(a)
    return (amin <= a <= amax) or (amin <= (180 - a) <= amax)

# Tìm 2 đầu trái-phải của CC trong mask
def cc_extreme_endpoints(mask_cc: np.ndarray):
    contours, _ = cv.findContours(mask_cc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours: return None, None
    pts = contours[0].reshape(-1, 2)
    iL, iR = np.argmin(pts[:,0]), np.argmax(pts[:,0])
    return tuple(pts[iL]), tuple(pts[iR])

# Giữ lại CC lớn nhất hoặc các CC có diện tích >= tỉ lệ so với CC lớn nhất
def keep_largest_or_by_ratio(mask: np.ndarray, keep_only_largest=True, min_area_ratio=0.1):
    fg = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)
    if num <= 1: return mask
    areas = stats[1:, cv.CC_STAT_AREA]
    max_area = areas.max()
    out = np.zeros_like(fg)
    for i in range(1, num):
        if keep_only_largest and stats[i, cv.CC_STAT_AREA] != max_area:
            continue
        if not keep_only_largest and stats[i, cv.CC_STAT_AREA] < max_area * min_area_ratio:
            continue
        out[labels == i] = 255
    return out

# ---------- main ----------
def main(img_path: str):
    # Đọc ảnh xám
    src = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Không mở được ảnh: {img_path}")
        return
    
    # Tiền xử lý: làm mờ + ngưỡng Otsu 
    blur = cv.GaussianBlur(src, (P["blur_k"], P["blur_k"]), 0)
    _, bin_inv = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # Kích thước ảnh
    h, w = bin_inv.shape[:2]
    base = min(h, w)

    # 1) TÌM VÀ LỌC CẠNH CHÉO
    L_open = max(25, int(P["len_ratio"] * base))
    accum_diag = np.zeros_like(bin_inv)
    for ang in range(0, 180, P["angle_step"]):
        if not in_diag(ang, P["diag_min_deg"], P["diag_max_deg"]):
            continue
        ker = line_kernel(L_open, P["thick"], ang)
        opened = cv.morphologyEx(bin_inv, cv.MORPH_OPEN, ker)
        accum_diag = cv.bitwise_or(accum_diag, opened)
    diag_mask = cv.morphologyEx(accum_diag, cv.MORPH_CLOSE, diskSE(P["close_gap"]))

    # Neo từ chéo (giãn nhẹ) để kiểm tra đầu mút ngang
    anchor_r = max(2, int(P["anchor_dilate_ratio"] * base))
    diag_anchor = cv.dilate(diag_mask, diskSE(anchor_r), iterations=1)

    # 2) TÌM VÀ LỌC CẠNH NGANG
    Lh = max(30, int(P["horiz_len_ratio"] * base))
    horiz_open = cv.morphologyEx(bin_inv, cv.MORPH_OPEN,
                                 cv.getStructuringElement(cv.MORPH_RECT, (Lh, 1)))

    # Lọc theo hình dạng + gating 2 đầu
    min_w = max(25, int(P["min_horiz_len_ratio"] * base))
    max_th = max(2,  int(P["max_horiz_thick_ratio"] * base))
    fgH = (horiz_open > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(fgH, 8)
    horiz_keep = np.zeros_like(fgH)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bw < min_w or bh > max_th:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        pL, pR = cc_extreme_endpoints(comp)
        if pL is None: 
            continue
        def touches_anchor(p):
            x0 = max(0, p[0]-anchor_r); x1 = min(w, p[0]+anchor_r+1)
            y0 = max(0, p[1]-anchor_r); y1 = min(h, p[1]+anchor_r+1)
            return cv.countNonZero(diag_anchor[y0:y1, x0:x1]) > 0
        if touches_anchor(pL) and touches_anchor(pR):
            horiz_keep = cv.bitwise_or(horiz_keep, comp)

    # 3) GHÉP + closing nhỏ
    merged = cv.bitwise_or(diag_mask, horiz_keep)
    clean = cv.morphologyEx(merged, cv.MORPH_CLOSE, diskSE(P["close_gap"]))

    # 4) **LOẠI nét bên trong**: giữ CC lớn nhất (vỏ sao)
    clean = keep_largest_or_by_ratio(
        clean,
        keep_only_largest=P["keep_only_largest_cc"],
        min_area_ratio=P["min_area_ratio"]
    )

    result = cv.bitwise_not(clean)

    # Hiển thị
    cv.imshow("Input", src)
    cv.imshow("Diagonal mask", diag_mask)
    cv.imwrite("diagonal_mask.png", diag_mask)
    cv.imshow("Accepted horizontals", horiz_keep)
    cv.imwrite("accepted_horizontals.png", horiz_keep)
    cv.imshow("Merged before CC filter", merged)
    cv.imwrite("merged_before_cc_filter.png", merged)
    cv.imshow("Result", result)
    cv.imwrite("star_result.png", result)
    cv.waitKey(0); cv.destroyAllWindows()

if __name__ == "__main__":
    # tìm file PNG trong cùng thư mục
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    png_files = [f for f in os.listdir(cur_dir) if f.lower().endswith(".png")]

    if len(png_files) == 0:
        print("Không tìm thấy file PNG nào trong thư mục!")
    else:
        img_path = os.path.join(cur_dir, "ex3.png")
        print(f"Đang xử lý file: {img_path}")
        main(img_path)



# Thứ tự thực hiện bao gồm các bước:
# 1. Đọc ảnh, lọc ngưỡng, sau đó lọc các cạnh chéo, thu được ảnh "Diagonal mask"
# 2. Tiếp tục lọc các cạnh ngang, thu được ảnh "Accepted horizontals"
# 3. Ghép 2 ảnh lại trước khi lọc CC, thu được ảnh "Merged before CC filter"
# 4. Cuối cùng, thực hiện lọc CC để loại bỏ nét thừa bên trong ngôi sao, thu được ảnh "Result".