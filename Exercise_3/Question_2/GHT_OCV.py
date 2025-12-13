import os
import cv2
import numpy as np

# ===============================
# PATHS
# ===============================
BASE = os.path.dirname(os.path.abspath(__file__))
PIC_DIR = os.path.join(BASE, "pictures")
TEMPLATE_PATH = os.path.join(PIC_DIR, "templates", "template_5.png")
SCENE_PATH = os.path.join(PIC_DIR, "aircraft_scene.png")

# Create result folder inside pictures
RESULT_DIR = os.path.join(PIC_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Output file named after template
tpl_name = os.path.splitext(os.path.basename(TEMPLATE_PATH))[0]
OUT_PATH = os.path.join(RESULT_DIR, f"{tpl_name}_result_OCV.png")

# ===============================
# PARAMETERS
# ===============================
ANGLE_STEP = 4
SCALE_MIN, SCALE_MAX = 0.5, 1.5
SCALE_STEPS = 6
CANNY_LOW = 40
CANNY_HIGH = 120
MIN_DIST = 40
VOTE_THRESH = 80
MAX_FINAL = 3

ROTATIONS = np.arange(0, 360, ANGLE_STEP)
SCALES = np.linspace(SCALE_MIN, SCALE_MAX, SCALE_STEPS)

# ===============================
# UTILITIES
# ===============================
def rotate_and_scale(img, angle, scale):
    """Rotate + scale around center."""
    h, w = img.shape[:2]
    cx, cy = w/2, h/2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)

    M[0,2] += nw/2 - cx
    M[1,2] += nh/2 - cy

    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    return out


def iou(a, b):
    """IoU của 2 box (x,y,w,h) dạng top-left."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh

    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, x2-x1), max(0, y2-y1)
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0


def nms(dets, thresh=0.3):
    """dets = list of dict: {'x','y','w','h','vote',...}"""
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: -d["vote"])
    keep = []

    for d in dets:
        ok = True
        for k in keep:
            if iou((d["x"], d["y"], d["w"], d["h"]),
                   (k["x"], k["y"], k["w"], k["h"])) > thresh:
                ok = False
                break
        if ok:
            keep.append(d)
        if len(keep) >= MAX_FINAL:
            break
    return keep

# ===============================
# MAIN – GHT BALLARD
# ===============================
def main():
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    scene_color = cv2.imread(SCENE_PATH)
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)

    if tpl0 is None:
        raise FileNotFoundError("Không đọc được template.")
    if scene_gray is None:
        raise FileNotFoundError("Không đọc được scene.")

    scene_edges = cv2.Canny(scene_gray, CANNY_LOW, CANNY_HIGH)

    # Create Ballard
    ght = cv2.createGeneralizedHoughBallard()
    ght.setCannyLowThresh(CANNY_LOW)
    ght.setCannyHighThresh(CANNY_HIGH)
    ght.setMinDist(MIN_DIST)
    ght.setVotesThreshold(VOTE_THRESH)

    Hs, Ws = scene_gray.shape
    detections = []

    print("Running GeneralizedHoughBallard with rotation + scale...")

    for scale in SCALES:
        for angle in ROTATIONS:

            # generate transformed template
            tpl_rs = rotate_and_scale(tpl0, angle, scale)
            th, tw = tpl_rs.shape[:2]
            if th < 5 or tw < 5: 
                continue
            if th >= Hs or tw >= Ws:
                continue

            tpl_edges = cv2.Canny(tpl_rs, CANNY_LOW, CANNY_HIGH)

            # Set template for Ballard
            ght.setTemplate(tpl_edges)

            # Detect → returns (positions, votes)
            pos, votes = ght.detect(scene_edges)

            if pos is None or len(pos) == 0:
                continue

            pos = np.array(pos).reshape(-1, 2)  # Ballard returns only (x,y)
            votes = np.array(votes).reshape(-1)

            for (x, y), v in zip(pos, votes):
                detections.append({
                    "x": int(x - tw//2),
                    "y": int(y - th//2),
                    "w": tw,
                    "h": th,
                    "vote": int(v),
                    "scale": float(scale),
                    "angle": float(angle)
                })

    # ===============================
    # Non-Maximum Suppression
    # ===============================
    filtered = nms(detections, thresh=0.3)

    # ===============================
    # Visualization
    # ===============================
    vis = scene_color.copy()
    for d in filtered:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis,
                    f"v={d['vote']} s={d['scale']:.2f} a={d['angle']}",
                    (x, max(0,y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)

    cv2.imwrite(OUT_PATH, vis)
    print("Saved result to:", OUT_PATH)

    try:
        cv2.imshow("GHT Ballard", vis)
        cv2.waitKey(0)
    except:
        pass


if __name__ == "__main__":
    main()
