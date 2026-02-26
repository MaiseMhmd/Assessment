"""
LSTM + OSNet Re-ID Tracking Evaluation
=======================================
Loads a pre-trained LSTM motion predictor + norm_stats, integrates OSNet x0.25
Re-ID for identity recovery, then evaluates on 50 test videos.

Produces:
  - tracking_results_lstm_reid.csv   (per-video + aggregate metrics)
  - output_videos/<video_id>_tracked.mp4  (3 rendered videos)

Configuration at the top of main() — edit paths before running.

Requirements:
    pip install ultralytics torchreid motmetrics opencv-python scipy
    (torchreid: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git)
"""

import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import motmetrics as mm

# ============================================================================
# CONFIG  (edit these)
# ============================================================================

TRAIN_DIR   = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\train")
ANNOT_FILE  = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\annotations_train.json")
MODELS_DIR  = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\hackathons and side projects\Technical assessment\saved_models")

OUTPUT_CSV  = Path("tracking_results_lstm_reid.csv")
OUTPUT_VID  = Path("output_videos")
OUTPUT_VID.mkdir(exist_ok=True)

# Test split
TEST_START  = 0
TEST_COUNT  = 50

# Videos to render  (folder name prefix match, e.g. "15e7e4d1")
VIDEO_IDS   = ["15e7e4d1", "033333fd", "18513251"]

# Tracker hyper-params
SEQ_LEN         = 10
MAX_AGE         = 30
MIN_HITS        = 3
IOU_THRESH_HIGH = 0.30
IOU_THRESH_LOW  = 0.15
EMA_ALPHA       = 0.7      # smoothing for motion prediction output
REID_WEIGHT     = 0.35     # blend ratio: (1-w)*iou_cost + w*reid_cost
REID_LONG_THRESH= 0.40     # cosine distance threshold for long-range re-ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# COLOURS
# ============================================================================
CLR_BLACK     = (0,   0,   0)
CLR_WHITE     = (255, 255, 255)
CLR_PREDICTED = (0,   165, 255)   # orange for occluded
_PAL = [(50,220,50),(80,160,255),(255,180,60),(180,80,255),
        (60,220,220),(255,80,160),(140,255,140),(255,140,80)]
def _col(tid): return _PAL[tid % len(_PAL)]


# ============================================================================
# NORMALIZATION STATS
# ============================================================================

class NormStats:
    def __init__(self):
        self.mean = np.zeros(4, dtype=np.float32)
        self.std  = np.ones(4,  dtype=np.float32)

    def normalize(self, feat):   return (feat - self.mean) / self.std
    def denormalize(self, feat): return feat * self.std + self.mean

    @classmethod
    def load(cls, path: Path):
        data = np.load(path)
        ns = cls()
        ns.mean = data["mean"]
        ns.std  = data["std"]
        print(f"  Loaded norm stats <- {path}")
        return ns


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, pred_len=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.pred_len    = pred_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden_size, input_size * pred_len)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).view(-1, self.pred_len, 4)


def load_lstm(path: Path, device: str) -> LSTMPredictor:
    model = LSTMPredictor()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    print(f"  Loaded LSTM       <- {path}")
    return model


# ============================================================================
# OSNET RE-ID
# ============================================================================

_OSNET      = None
_OSNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_OSNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_osnet():
    global _OSNET
    if _OSNET is not None:
        return _OSNET
    try:
        import torchreid
    except ImportError:
        raise ImportError(
            "torchreid not installed.\n"
            "Run: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
        )
    print("  Initialising OSNet x0.25 (downloads weights on first run)...")
    _OSNET = torchreid.models.build_model(
        name="osnet_x0_25", num_classes=1000, pretrained=True
    )
    _OSNET.eval().to(DEVICE)
    print("  ✔  OSNet ready.")
    return _OSNET


def extract_reid(img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = cv2.resize(img[y1:y2, x1:x2], (128, 256))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = (crop.astype(np.float32) / 255.0 - _OSNET_MEAN) / _OSNET_STD
    t    = torch.FloatTensor(crop).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = _get_osnet()(t)[0].cpu().numpy().astype(np.float32)
    n = np.linalg.norm(f)
    return f / (n + 1e-8)


def reid_dist(a, b) -> float:
    if a is None or b is None:
        return 0.5
    return float((1.0 - float(np.dot(a, b))) / 2.0)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    bbox:       np.ndarray
    confidence: float
    class_id:   int
    frame_id:   int = 0
    reid_feat:  Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Track:
    id:               int
    bbox:             np.ndarray
    class_id:         int
    confidence:       float
    trajectory:       List[np.ndarray]
    hits:             int             = 0
    age:              int             = 0
    time_since_update:int             = 0
    is_predicted:     bool            = False
    reid_feat:        Optional[np.ndarray] = field(default=None, repr=False)
    _mp:              object          = field(default=None, repr=False)


# ============================================================================
# MOTION PREDICTOR WRAPPER
# ============================================================================

class MotionPredictor:
    def __init__(self, model: nn.Module, norm_stats: NormStats):
        self.model      = model
        self.norm_stats = norm_stats
        self._hist:   List[np.ndarray]      = []
        self._smooth: Optional[np.ndarray]  = None

    def update(self, bbox: np.ndarray):
        self._hist.append(bbox.copy())
        if len(self._hist) > SEQ_LEN:
            self._hist = self._hist[-SEQ_LEN:]
        self._smooth = None          # invalidate EMA on new observation

    def predict(self) -> np.ndarray:
        raw = self._raw_predict()
        if self._smooth is None:
            self._smooth = raw
        else:
            self._smooth = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * self._smooth
        return self._smooth.copy()

    def _raw_predict(self) -> np.ndarray:
        if not self._hist:
            return np.zeros(4, dtype=np.float32)
        if len(self._hist) < SEQ_LEN:
            return self._hist[-1].copy()

        feats = []
        for bbox in self._hist[-SEQ_LEN:]:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w  =  bbox[2] - bbox[0]
            h  =  bbox[3] - bbox[1]
            feats.append([cx, cy, w, h])

        normed = self.norm_stats.normalize(np.array(feats, dtype=np.float32))
        inp    = torch.FloatTensor(normed).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_norm = self.model(inp)[0, 0].cpu().numpy()

        cx, cy, w, h = self.norm_stats.denormalize(out_norm)
        w = max(w, 1.0); h = max(h, 1.0)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)


# ============================================================================
# NEURAL TRACKER  (IoU + Re-ID + long-range re-identification)
# ============================================================================

class NeuralTracker:
    def __init__(self, model: nn.Module, norm_stats: NormStats):
        self.model      = model
        self.norm_stats = norm_stats
        self.tracks:    List[Track] = []
        self.next_id    = 1
        self._lost_buf: List[Track] = []   # recently-lost confirmed tracks

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection],
               frame: Optional[np.ndarray] = None) -> List[Track]:

        # 0. Extract Re-ID features
        if frame is not None:
            for d in detections:
                if d.reid_feat is None:
                    d.reid_feat = extract_reid(frame, d.bbox)

        # 1. Predict positions for all active tracks
        self._predict_all()

        # 2. Two-stage matching (high-conf first, low-conf second)
        high = [d for d in detections if d.confidence >= 0.5]
        low  = [d for d in detections if d.confidence <  0.5]
        all_trk_idx = list(range(len(self.tracks)))

        m1, unm_d1, unm_t1 = self._match(high, all_trk_idx, IOU_THRESH_HIGH)
        m2, _unm_d2, _      = self._match(low,  unm_t1,     IOU_THRESH_LOW)

        matched_trk_idx = set()
        for di, ti in m1:
            self._apply(self.tracks[ti], high[di], frame)
            matched_trk_idx.add(ti)
        for di, ti in m2:
            self._apply(self.tracks[ti], low[di], frame)
            matched_trk_idx.add(ti)

        # 3. Age unmatched tracks
        for ti in range(len(self.tracks)):
            if ti not in matched_trk_idx:
                self.tracks[ti].time_since_update += 1
                self.tracks[ti].is_predicted = True

        # 4. Prune dead tracks → move confirmed ones to lost buffer
        alive, dying = [], []
        for t in self.tracks:
            (dying if t.time_since_update >= MAX_AGE else alive).append(t)
        for t in dying:
            if t.hits >= MIN_HITS and t.reid_feat is not None:
                self._lost_buf.append(t)
        self._lost_buf = self._lost_buf[-60:]
        self.tracks    = alive

        # 5. Re-identify or create new tracks for unmatched high-conf dets
        used_in_m1 = {di for di, _ in m1}
        for di in unm_d1:
            if di in used_in_m1:
                continue
            det = high[di]
            recovered = self._reidentify(det)
            if recovered is not None:
                self._apply(recovered, det, frame)
                recovered.time_since_update = 0
                recovered.is_predicted      = False
                self.tracks.append(recovered)
                self._lost_buf = [t for t in self._lost_buf
                                  if t.id != recovered.id]
            else:
                self._new_track(det)

        return [t for t in self.tracks if t.hits >= MIN_HITS]

    # ------------------------------------------------------------------
    def _predict_all(self):
        for t in self.tracks:
            if t._mp is None:
                t._mp = MotionPredictor(self.model, self.norm_stats)
                for b in t.trajectory:
                    t._mp.update(b)
            t.bbox = t._mp.predict()

    def _match(self, dets, trk_indices, iou_thr):
        if not dets or not trk_indices:
            return [], list(range(len(dets))), list(trk_indices)

        cost = np.ones((len(dets), len(trk_indices)), dtype=np.float32)
        for di, det in enumerate(dets):
            for j, ti in enumerate(trk_indices):
                iou = self._iou(det.bbox, self.tracks[ti].bbox)
                if iou < 0.05:
                    continue
                cost[di, j] = (
                    (1 - REID_WEIGHT) * (1 - iou)
                    + REID_WEIGHT * reid_dist(det.reid_feat,
                                              self.tracks[ti].reid_feat)
                )

        ri, ci = linear_sum_assignment(cost)
        matched, used_d, used_t = [], set(), set()
        for r, c in zip(ri, ci):
            ti = trk_indices[c]
            if self._iou(dets[r].bbox, self.tracks[ti].bbox) >= iou_thr:
                matched.append((r, ti))
                used_d.add(r)
                used_t.add(ti)

        return (matched,
                [i  for i  in range(len(dets))   if i  not in used_d],
                [ti for ti in trk_indices         if ti not in used_t])

    def _reidentify(self, det) -> Optional[Track]:
        if det.reid_feat is None or not self._lost_buf:
            return None
        best_d, best_t = REID_LONG_THRESH, None
        for t in self._lost_buf:
            d = reid_dist(det.reid_feat, t.reid_feat)
            if d < best_d:
                best_d, best_t = d, t
        return best_t

    def _apply(self, trk: Track, det: Detection,
               frame: Optional[np.ndarray]):
        trk.bbox       = det.bbox.copy()
        trk.confidence = det.confidence
        trk.trajectory.append(det.bbox.copy())
        trk.hits += 1
        trk.time_since_update = 0
        trk.is_predicted      = False
        if trk._mp is None:
            trk._mp = MotionPredictor(self.model, self.norm_stats)
        trk._mp.update(det.bbox)
        # EMA update of appearance model
        if det.reid_feat is not None:
            if trk.reid_feat is None:
                trk.reid_feat = det.reid_feat.copy()
            else:
                trk.reid_feat = 0.8 * trk.reid_feat + 0.2 * det.reid_feat
                n = np.linalg.norm(trk.reid_feat)
                trk.reid_feat /= (n + 1e-8)

    def _new_track(self, det: Detection):
        t = Track(
            id=self.next_id, bbox=det.bbox.copy(),
            class_id=det.class_id, confidence=det.confidence,
            trajectory=[det.bbox.copy()], hits=1,
            reid_feat=det.reid_feat,
        )
        t._mp = MotionPredictor(self.model, self.norm_stats)
        t._mp.update(det.bbox)
        self.tracks.append(t)
        self.next_id += 1

    @staticmethod
    def _iou(a, b) -> float:
        xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
        xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
        inter = max(0., xi2 - xi1) * max(0., yi2 - yi1)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return float(inter / u) if u > 0 else 0.


# ============================================================================
# DRAWING HELPERS
# ============================================================================

def _dashed_rect(img, p1, p2, color, thick=2, dash=10):
    x1, y1 = p1; x2, y2 = p2
    for sx,sy,ex,ey in [(x1,y1,x2,y1),(x2,y1,x2,y2),
                        (x2,y2,x1,y2),(x1,y2,x1,y1)]:
        dx, dy = ex-sx, ey-sy
        L = max(1, int(np.hypot(dx, dy)))
        steps = L // (dash*2) + 1
        for s in range(steps):
            t0 = min(1., s*2*dash/L)
            t1 = min(1., (s*2+1)*dash/L)
            cv2.line(img,
                     (int(sx+t0*dx), int(sy+t0*dy)),
                     (int(sx+t1*dx), int(sy+t1*dy)), color, thick)


def _lbl(img, txt, x, y):
    f  = cv2.FONT_HERSHEY_SIMPLEX; sc, th = 0.5, 1
    (tw, fh), bl = cv2.getTextSize(txt, f, sc, th)
    cv2.rectangle(img, (x, max(0, y-fh-4)), (x+tw+6, y+bl), CLR_BLACK, -1)
    cv2.putText(img, txt, (x+3, y), f, sc, CLR_WHITE, th)


def draw_frame(img: np.ndarray, tracks: List[Track],
               label: str = "") -> np.ndarray:
    out = img.copy(); H, W = out.shape[:2]
    for t in tracks:
        x1 = max(0, int(t.bbox[0])); y1 = max(0, int(t.bbox[1]))
        x2 = min(W-1, int(t.bbox[2])); y2 = min(H-1, int(t.bbox[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        if t.is_predicted:
            _dashed_rect(out, (x1,y1), (x2,y2), CLR_PREDICTED)
            _lbl(out, f"ID:{t.id} OCCLUDED", x1, y1)
        else:
            cv2.rectangle(out, (x1,y1), (x2,y2), _col(t.id), 2)
            _lbl(out, f"ID:{t.id}  {t.confidence:.2f}", x1, y1)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX; sc, th = 0.65, 2
        (tw, fh), bl = cv2.getTextSize(label, font, sc, th)
        wx, wy = 10, H - 10
        cv2.rectangle(out, (wx-4, wy-fh-6), (wx+tw+4, wy+bl+2), CLR_BLACK, -1)
        cv2.putText(out, label, (wx, wy), font, sc, CLR_WHITE, th)
    return out


# ============================================================================
# METRICS
# ============================================================================

def compute_ade(pred_trajs, gt_trajs) -> float:
    if not pred_trajs or not gt_trajs:
        return 0.0
    errors = []
    for pt in pred_trajs:
        if not pt: continue
        best = float("inf")
        for gt in gt_trajs:
            if not gt: continue
            n  = min(len(pt), len(gt))
            pc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in pt[:n]])
            gc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in gt[:n]])
            err = np.mean(np.linalg.norm(pc - gc, axis=1))
            best = min(best, err)
        if best != float("inf"):
            errors.append(best)
    return float(np.mean(errors)) if errors else 0.0


def evaluate_video(predictions, annotations, video_folder_name):
    video_info = None
    for video in annotations["videos"]:
        folder = video["file_names"][0].replace("\\", "/").split("/")[0]
        if folder == video_folder_name:
            video_info = video; break
    if video_info is None:
        return None

    vid_id  = video_info["id"]
    vid_len = video_info.get("length",
                             len(video_info.get("file_names", [])))

    gt_by_frame = defaultdict(list)
    for ann in annotations["annotations"]:
        if ann["video_id"] != vid_id: continue
        tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        for fi, bbox in enumerate(ann.get("bboxes", [])):
            if not bbox or len(bbox) != 4: continue
            x, y, w, h = bbox
            if w > 0 and h > 0:
                gt_by_frame[fi].append({"id": tid,
                                        "bbox": [x, y, x+w, y+h]})

    acc        = mm.MOTAccumulator(auto_id=True)
    pred_trajs = defaultdict(list)
    gt_trajs   = defaultdict(list)

    for pf in predictions:
        fid = pf["frame_id"]
        if fid >= vid_len: continue
        gt_objs   = gt_by_frame.get(fid, [])
        gt_ids    = [o["id"]   for o in gt_objs]
        gt_bbs    = [o["bbox"] for o in gt_objs]
        pred_ids  = [t[0] for t in pf["tracks"]]
        pred_bbs  = [t[1] for t in pf["tracks"]]

        for tid, bbox in zip(pred_ids, pred_bbs):   pred_trajs[tid].append(bbox)
        for tid, bbox in zip(gt_ids,   gt_bbs):     gt_trajs[tid].append(bbox)

        if gt_bbs and pred_bbs:
            dist = np.array([
                [1 - NeuralTracker._iou(np.array(g), np.array(p))
                 for p in pred_bbs]
                for g in gt_bbs
            ])
        else:
            dist = np.empty((len(gt_bbs), len(pred_bbs)))

        acc.update(gt_ids, pred_ids, dist)

    mh      = mm.metrics.create()
    summary = mh.compute(acc,
                         metrics=["mota", "num_switches", "idf1"],
                         name="acc")
    ade = compute_ade(list(pred_trajs.values()),
                      list(gt_trajs.values()))

    return {
        "mota":        float(summary["mota"].values[0])
                       if "mota"         in summary.columns else 0.0,
        "idf1":        float(summary["idf1"].values[0])
                       if "idf1"         in summary.columns else 0.0,
        "id_switches": int(summary["num_switches"].values[0])
                       if "num_switches" in summary.columns else 0,
        "ade":         ade,
    }


def aggregate(video_metrics):
    avg_mota = float(np.mean([m["mota"] for m in video_metrics]))
    avg_idf1 = float(np.mean([m["idf1"] for m in video_metrics]))
    avg_ade  = float(np.mean([m["ade"]  for m in video_metrics]))
    total_sw = int(np.sum( [m["id_switches"] for m in video_metrics]))
    return {
        "mota_norm":   round(max(0., min(1., avg_mota)), 4),
        "idf1_norm":   round(max(0., min(1., avg_idf1)), 4),
        "ade_norm":    round(1.0 - min(1., avg_ade / 200.0), 4),
        "avg_ade_px":  round(avg_ade,  2),
        "id_switches": total_sw,
    }


# ============================================================================
# VIDEO WRITER HELPER
# ============================================================================

def write_video(frames_dir: Path, tracks_by_frame: dict,
                out_path: Path, video_id: str,
                label: str = "LSTM+ReID"):
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"  [warn] no frames in {frames_dir}")
        return

    sample = cv2.imread(str(frames[0]))
    H, W   = sample.shape[:2]
    fps    = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    for fi, fp in enumerate(frames):
        img    = cv2.imread(str(fp))
        if img is None: continue
        tracks = tracks_by_frame.get(fi, [])
        vis    = draw_frame(img, tracks,
                            label=f"{video_id}_{label}")
        vw.write(vis)

    vw.release()
    print(f"  Video saved -> {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Device: {DEVICE}")

    # ── Load model & stats ────────────────────────────────────────────
    norm_stats = NormStats.load(MODELS_DIR / "norm_stats.npz")
    lstm_model = load_lstm(MODELS_DIR / "lstm_model.pth", DEVICE)

    # Warm-up OSNet (downloads weights if needed)
    print("  Warming up OSNet...")
    _get_osnet()

    # ── Load annotations ─────────────────────────────────────────────
    print(f"  Loading annotations from {ANNOT_FILE} ...")
    with open(ANNOT_FILE) as f:
        annotations = json.load(f)

    # ── Test video folders ───────────────────────────────────────────
    all_folders  = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    test_folders = all_folders[TEST_START : TEST_START + TEST_COUNT]
    print(f"  Test videos: {len(test_folders)}  "
          f"(idx {TEST_START}–{TEST_START+TEST_COUNT-1})")

    detector = YOLO("yolo11n.pt")

    # ── Evaluate ─────────────────────────────────────────────────────
    video_metrics = []
    per_video_rows = []

    for vi, vfolder in enumerate(test_folders, 1):
        vname  = vfolder.name
        frames = sorted(vfolder.glob("*.jpg"))
        print(f"  [{vi:02d}/{len(test_folders)}] {vname}  ({len(frames)} frames)")
        if not frames:
            continue

        tracker         = NeuralTracker(lstm_model, norm_stats)
        predictions     = []
        tracks_by_frame = {}        # for video rendering

        for fi, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None: continue

            dets = []
            for result in detector(img, conf=0.3, verbose=False):
                for i in range(len(result.boxes)):
                    dets.append(Detection(
                        bbox=result.boxes.xyxy[i].cpu().numpy(),
                        confidence=float(result.boxes.conf[i].cpu().numpy()),
                        class_id=int(result.boxes.cls[i].cpu().numpy()),
                        frame_id=fi,
                    ))

            active = tracker.update(dets, frame=img)
            predictions.append({
                "frame_id": fi,
                "tracks":   [(t.id, t.bbox.tolist(),
                              t.class_id, t.confidence)
                             for t in active],
            })
            tracks_by_frame[fi] = [
                Track(id=t.id, bbox=t.bbox.copy(),
                      class_id=t.class_id, confidence=t.confidence,
                      trajectory=[], hits=t.hits,
                      is_predicted=t.is_predicted)
                for t in active
            ]

        # Per-video metrics
        m = evaluate_video(predictions, annotations, vname)
        if m:
            video_metrics.append(m)
            per_video_rows.append({"video": vname, **m})
            print(f"       MOTA={m['mota']:.4f}  IDF1={m['idf1']:.4f}  "
                  f"ADE={m['ade']:.1f}px  ID-SW={m['id_switches']}")

        # Render video if requested
        if any(vname.startswith(vid) for vid in VIDEO_IDS):
            out_mp4 = OUTPUT_VID / f"{vname}_tracked.mp4"
            write_video(vfolder, tracks_by_frame, out_mp4, vname)

    # ── Aggregate ─────────────────────────────────────────────────────
    if video_metrics:
        agg = aggregate(video_metrics)
        print("\n" + "="*60)
        print("AGGREGATE RESULTS  (LSTM + OSNet Re-ID)")
        print("="*60)
        print(f"  MOTA (norm):   {agg['mota_norm']}")
        print(f"  IDF1 (norm):   {agg['idf1_norm']}")
        print(f"  ADE  (norm):   {agg['ade_norm']}")
        print(f"  ADE  (px):     {agg['avg_ade_px']}")
        print(f"  ID Switches:   {agg['id_switches']}")

        # ── Save CSV ─────────────────────────────────────────────────
        rows = per_video_rows + [{"video": "AGGREGATE", **agg}]
        df   = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n  Results saved -> {OUTPUT_CSV}")
        print(df.to_string(index=False))

    print(f"\nDone.  Rendered videos (if any) -> {OUTPUT_VID.resolve()}")


if __name__ == "__main__":
    main()