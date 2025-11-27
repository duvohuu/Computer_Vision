import os, cv2, numpy as np
from scipy.ndimage import maximum_filter

# ===============================
# PATH
# ===============================
BASE = os.path.dirname(os.path.abspath(__file__))
PIC_DIR = os.path.join(BASE, "pictures")
TEMPLATE_PATH = os.path.join(PIC_DIR, "templates", "template_4.png")
SCENE_PATH = os.path.join(PIC_DIR, "aircraft_scene.png")

# Create result folder inside pictures
RESULT_DIR = os.path.join(PIC_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Output file named after template
tpl_name = os.path.splitext(os.path.basename(TEMPLATE_PATH))[0]
OUT_PATH = os.path.join(RESULT_DIR, f"{tpl_name}_result.png")

# ===============================
# PARAMETERS
# ===============================
CANNY_LOW, CANNY_HIGH = 40, 120
ANGLE_BIN = 8               
ACC_REL_THRESH = 0.2        
MAX_PEAKS = 3            

# rotation & scale search space
ROT_STEP = 5             
SCALE_MIN, SCALE_MAX = 0.5, 1.5
SCALE_STEPS = 6                  

ROTATIONS = np.arange(0, 360, ROT_STEP)
SCALES = np.linspace(SCALE_MIN, SCALE_MAX, SCALE_STEPS)

# Pre-compute rotation matrices
ROT_RAD = np.radians(ROTATIONS)
ROT_COS = np.cos(ROT_RAD)
ROT_SIN = np.sin(ROT_RAD)

# ===============================
# BUILD R-TABLE (Pure Vectorized)
# ===============================
def build_rtable(template_gray):
    _, bin_img = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(bin_img, CANNY_LOW, CANNY_HIGH)

    gx = cv2.Sobel(template_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(template_gray, cv2.CV_32F, 0, 1, ksize=3)
    
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        raise ValueError("Template rỗng sau khi nhị phân hóa.")
    cx, cy = xs.mean(), ys.mean()

    ys_e, xs_e = np.where(edges > 0)
    
    # Vectorized: tính toán tất cả cùng lúc
    theta = np.arctan2(gy[ys_e, xs_e], gx[ys_e, xs_e])
    theta = np.degrees(theta) % 360.0
    bin_ang = (theta // ANGLE_BIN).astype(int) * ANGLE_BIN

    rx = cx - xs_e
    ry = cy - ys_e
    
    # Tạo angle bins array
    n_bins = 360 // ANGLE_BIN
    rtable_bins = np.arange(0, 360, ANGLE_BIN)
    
    # Mỗi bin lưu indices của các edge points thuộc bin đó
    rtable_idx = {}
    for b in rtable_bins:
        mask = bin_ang == b
        if np.any(mask):
            rtable_idx[b] = np.where(mask)[0]
    
    # Lưu toàn bộ r vectors
    r_vectors = np.column_stack([rx, ry])  # shape: (N_edges, 2)
    
    return rtable_idx, r_vectors, (cx, cy)

# ===============================
# VOTING (FULLY VECTORIZED - NO NESTED LOOPS!)
# ===============================
def vote_with_rot_scale_vectorized(scene_gray, rtable_idx, r_vectors):
    edges = cv2.Canny(scene_gray, CANNY_LOW, CANNY_HIGH)
    gx = cv2.Sobel(scene_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(scene_gray, cv2.CV_32F, 0, 1)

    H, W = scene_gray.shape
    n_rots = len(ROTATIONS)
    n_scales = len(SCALES)

    # Tạo 4D accumulator: [H, W, n_scales, n_rots]
    acc_4d = np.zeros((H, W, n_scales, n_rots), dtype=np.int32)

    ys_scene, xs_scene = np.where(edges > 0)
    n_scene = len(xs_scene)
    
    if n_scene == 0:
        return acc_4d

    # Scene angles cho TẤT CẢ edge pixels
    theta_scene = np.arctan2(gy[ys_scene, xs_scene], gx[ys_scene, xs_scene])
    theta_scene = np.degrees(theta_scene) % 360.0  # shape: (n_scene,)
    
    print(f"Processing {n_scene} scene edges...")

    # Strategy: Xử lý theo batches của (rotation, scale) combinations
    
    for rot_idx in range(n_rots):
        phi = ROTATIONS[rot_idx]
        cos_phi = ROT_COS[rot_idx]
        sin_phi = ROT_SIN[rot_idx]
        
        # Template angles tương ứng với scene angles khi rotate phi
        theta_tpl = (theta_scene - phi) % 360.0  # (n_scene,)
        bin_tpl = ((theta_tpl // ANGLE_BIN).astype(int) * ANGLE_BIN) % 360
        
        # Rotate TẤT CẢ r_vectors cùng lúc
        # r_rotated shape: (n_template_edges, 2)
        r_rotated = np.empty_like(r_vectors)
        r_rotated[:, 0] = r_vectors[:, 0] * cos_phi - r_vectors[:, 1] * sin_phi
        r_rotated[:, 1] = r_vectors[:, 0] * sin_phi + r_vectors[:, 1] * cos_phi
        
        for scale_idx, scale in enumerate(SCALES):
            # Scale TẤT CẢ rotated vectors
            r_scaled = r_rotated * scale  # (n_template_edges, 2)
            
            # Với mỗi bin, vote TẤT CẢ combinations cùng lúc
            for bin_val, template_indices in rtable_idx.items():
                # Lấy scene edges thuộc bin này
                scene_mask = bin_tpl == bin_val
                scene_indices = np.where(scene_mask)[0]
                
                if len(scene_indices) == 0:
                    continue
                
                # Lấy tọa độ scene edges này
                xs_s = xs_scene[scene_indices]  # (n_s,)
                ys_s = ys_scene[scene_indices]  # (n_s,)
                
                # Lấy r vectors từ template cho bin này
                r_bin = r_scaled[template_indices]  # (n_t, 2)
                
                # VECTORIZED OUTER SUM:
                # Mỗi scene edge (xs_s[i], ys_s[i]) + mỗi r_bin[j]
                # Tạo mesh: (n_s, n_t, 2)
                xs_mesh = xs_s[:, None] + r_bin[:, 0][None, :]  # (n_s, n_t)
                ys_mesh = ys_s[:, None] + r_bin[:, 1][None, :]  # (n_s, n_t)
                
                # Round và flatten
                xi = np.round(xs_mesh).astype(int).ravel()
                yi = np.round(ys_mesh).astype(int).ravel()
                
                # Filter valid
                valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                xi = xi[valid]
                yi = yi[valid]
                
                # Vote bằng np.add.at (atomic add)
                np.add.at(acc_4d, (yi, xi, scale_idx, rot_idx), 1)
    
    return acc_4d

# ===============================
# FIND PEAKS (Vectorized với scipy)
# ===============================
def find_peaks_vectorized(acc_4d):
    max_val = acc_4d.max()
    if max_val == 0:
        return []
    
    # Normalize
    acc_norm = acc_4d.astype(np.float32) / max_val
    
    # Apply threshold
    acc_thresh = acc_norm * (acc_norm >= ACC_REL_THRESH)
    
    # Non-maximum suppression trong 4D
    # Áp dụng local maximum filter
    footprint = np.ones((5, 5, 1, 1))  # spatial NMS only
    local_max = maximum_filter(acc_thresh, footprint=footprint, mode='constant')
    
    # Peaks là nơi giá trị = local max
    peaks_mask = (acc_thresh == local_max) & (acc_thresh > 0)
    
    # Lấy tọa độ peaks
    coords = np.argwhere(peaks_mask)  # (N_peaks, 4)
    scores = acc_thresh[peaks_mask]
    
    # Sort theo score
    sort_idx = np.argsort(-scores)[:MAX_PEAKS]
    
    peaks = []
    for idx in sort_idx:
        y, x, scale_idx, rot_idx = coords[idx]
        score = scores[idx]
        peaks.append((int(x), int(y), float(score), 
                     float(SCALES[scale_idx]), int(ROTATIONS[rot_idx])))
    
    return peaks

# ===============================
# MAIN
# ===============================
def main():
    import time
    
    tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    scene = cv2.imread(SCENE_PATH)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    print("Building R-table...")
    t0 = time.time()
    rtable_idx, r_vectors, ref = build_rtable(tpl)
    t1 = time.time()
    print(f"R-table: {len(rtable_idx)} bins, {len(r_vectors)} vectors in {t1-t0:.3f}s")

    print("Voting (vectorized)...")
    t0 = time.time()
    acc_4d = vote_with_rot_scale_vectorized(scene_gray, rtable_idx, r_vectors)
    t1 = time.time()
    print(f"Voting done in {t1-t0:.3f}s")
    
    print("Finding peaks...")
    t0 = time.time()
    peaks = find_peaks_vectorized(acc_4d)
    t1 = time.time()
    print(f"Found {len(peaks)} peaks in {t1-t0:.3f}s")

    # Visualize
    vis = scene.copy()
    for (x, y, s, scale, rot) in peaks:
        cv2.circle(vis, (int(x), int(y)), 20, (0, 255, 0), 2)
        cv2.putText(vis, f"{s:.2f} s={scale:.2f} r={rot}deg", (int(x)-40, int(y)-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imwrite(OUT_PATH, vis)
    print("Saved to:", OUT_PATH)

    try:
        window = "GHT Result (Fully Vectorized)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        fallback = OUT_PATH.replace(".png", "_show.png")
        cv2.imwrite(fallback, vis)
        print("Không thể mở GUI:", e)


if __name__ == "__main__":
    main()