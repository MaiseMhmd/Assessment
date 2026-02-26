"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          OVIS  —  LSTM Tracker  (Improved / Refined)                        ║
║                                                                              ║
║  What's new vs baseline                                                      ║
║  ──────────────────────                                                      ║
║  • Input: [cx,cy,w,h,vx,vy]  (6-dim, velocity-aware)                        ║
║  • 30-frame history window   (was 10)                                        ║
║  • Huber loss                (robust to noisy/outlier bboxes)                ║
║  • LayerNorm + dropout       (stable training)                               ║
║  • AdamW + cosine schedule   (better generalisation)                         ║
║  • Data augmentation         (noise + frame dropout during training)         ║
║  • EMA smoothing             (stable predicted bbox)                         ║
║  • Hungarian matching        (globally optimal assignment)                   ║
║  • Two-stage matching        (high-conf → low-conf rescue)                   ║
║  • OSNet ReID                (appearance embedding every frame +             ║
║                               re-association after long occlusion)           ║
║                                                                              ║
║  Outputs                                                                     ║
║  ───────                                                                     ║
║  saved_models/lstm_improved.pth                                              ║
║  saved_models/norm_stats_improved.npz                                        ║
║  refined_model/15e7e4d1_LSTM_Improved.mp4                                   ║
║  refined_model/033333fd_LSTM_Improved.mp4                                   ║
║  refined_model/18513251_LSTM_Improved.mp4                                   ║
║  tracking_results_lstm_improved.csv                                          ║
║                                                                              ║
║  Install once:                                                               ║
║    pip install torchreid ultralytics motmetrics scipy                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from ultralytics import YOLO
import motmetrics as mm


# ============================================================================
# CONFIG
# ============================================================================

TRAIN_DIR  = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\train")
ANNOT_FILE = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\annotations_train.json")
MODELS_DIR = Path("saved_models")
VIDEO_DIR  = Path("refined_model")      # ← all 3 output videos land here
OUTPUT_CSV = Path("tracking_results_lstm_improved.csv")

# Videos to render
VIDEO_IDS = ["15e7e4d1", "033333fd", "18513251"]

# Training split
TRAIN_START = 50
TRAIN_COUNT = 300
TEST_START  = 0
TEST_COUNT  = 10                        # evaluate metrics on 10 videos

# LSTM architecture
SEQ_LEN     = 30
PRED_LEN    = 1
INPUT_DIM   = 6                         # [cx, cy, w, h, vx, vy]
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.2

# Training
EPOCHS      = 25
BATCH_SIZE  = 64
LR          = 3e-4

# Tracker
MAX_AGE          = 40
MIN_HITS         = 2
IOU_THRESH_HIGH  = 0.4
IOU_THRESH_LOW   = 0.15
REID_WEIGHT      = 0.40                 # 0 = pure IoU, 1 = pure ReID
REID_LONG_THRESH = 0.45                 # max cosine dist for re-ID recovery
EMA_ALPHA        = 0.60                 # prediction smoothing

# Video
FPS_OUT = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BGR colours
CLR_DETECTED  = (50,  220,  50)
CLR_PREDICTED = (30,   30, 220)
CLR_WHITE     = (255, 255, 255)
CLR_BLACK     = (0,   0,   0)


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
    id:                int
    bbox:              np.ndarray
    class_id:          int
    confidence:        float
    trajectory:        List[np.ndarray] = field(default_factory=list)
    hits:              int = 0
    time_since_update: int = 0
    is_predicted:      bool = False
    reid_feat:         Optional[np.ndarray] = field(default=None, repr=False)
    _mp:               object = field(default=None, repr=False)


# ============================================================================
# NORMALIZATION  (6-dim)
# ============================================================================

class NormStats:
    def __init__(self, dim: int = INPUT_DIM):
        self.dim  = dim
        self.mean = np.zeros(dim, dtype=np.float32)
        self.std  = np.ones(dim,  dtype=np.float32)

    def fit(self, trajectories: List[List]):
        all_feats = []
        for traj in trajectories:
            all_feats.extend(_to_vel_features(traj))
        arr       = np.array(all_feats, dtype=np.float32)
        self.mean = arr.mean(axis=0)
        self.std  = arr.std(axis=0) + 1e-8

    def normalize(self, x):   return (x - self.mean) / self.std
    def denormalize(self, x): return x * self.std + self.mean

    def save(self, path: Path):
        np.savez(str(path), mean=self.mean, std=self.std, dim=np.array([self.dim]))
        print(f"  Saved norm stats  → {path}")

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d  = np.load(str(path))
        ns = cls(dim=int(d["dim"][0]))
        ns.mean = d["mean"].astype(np.float32)
        ns.std  = d["std"].astype(np.float32)
        print(f"  Loaded norm stats ← {path}")
        return ns


# ============================================================================
# VELOCITY FEATURES
# ============================================================================

def _to_vel_features(bboxes: List) -> List[List[float]]:
    """[x1,y1,x2,y2] list  →  [cx,cy,w,h,vx,vy] list."""
    feats = []
    pcx = pcy = None
    for b in bboxes:
        cx = (b[0]+b[2])/2.0;  cy = (b[1]+b[3])/2.0
        w  =  b[2]-b[0];       h  =  b[3]-b[1]
        vx = (cx-pcx) if pcx is not None else 0.0
        vy = (cy-pcy) if pcy is not None else 0.0
        feats.append([cx, cy, w, h, vx, vy])
        pcx, pcy = cx, cy
    return feats


# ============================================================================
# DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, norm_stats: NormStats,
                 seq_len=SEQ_LEN, pred_len=PRED_LEN):
        self.norm_stats = norm_stats
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.samples: List[dict] = []

        for traj in sequences:
            if len(traj) < seq_len + pred_len:
                continue
            raw = _to_vel_features(traj)
            for i in range(len(raw) - seq_len - pred_len + 1):
                inp = np.array(raw[i           : i+seq_len],          dtype=np.float32)
                tgt = np.array(raw[i+seq_len   : i+seq_len+pred_len], dtype=np.float32)
                self.samples.append({
                    "input":  norm_stats.normalize(inp),
                    "target": norm_stats.normalize(tgt),
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        inp = self.samples[idx]["input"].copy()
        tgt = self.samples[idx]["target"]

        # ── Augmentation ──────────────────────────────────────────────────
        noise = np.random.randn(*inp.shape).astype(np.float32) * 0.03
        noise[:, 2:4] *= 0.4        # less noise on w, h
        inp += noise
        if np.random.rand() < 0.35:
            drop = np.random.randint(0, len(inp))
            inp[drop] = inp[max(0, drop-1)]

        return torch.FloatTensor(inp), torch.FloatTensor(tgt)


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMImproved(nn.Module):
    def __init__(self, input_size=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, pred_len=PRED_LEN, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.pred_len    = pred_len

        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 4 * pred_len),    # predicts [cx, cy, w, h]
        )

    def forward(self, x):
        x      = self.input_norm(x)
        h0     = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0     = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out    = self.drop(out[:, -1, :])
        return self.head(out).view(-1, self.pred_len, 4)


def save_model(model: nn.Module, path: Path):
    torch.save(model.state_dict(), str(path))
    print(f"  Saved  → {path}")


def load_model_weights(model: nn.Module, path: Path) -> nn.Module:
    model.load_state_dict(torch.load(str(path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded ← {path}")
    return model


# ============================================================================
# OSNET  RE-ID
# ============================================================================

_OSNET = None
_OSNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_OSNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_osnet():
    global _OSNET
    if _OSNET is not None:
        return _OSNET
    try:
        import torchreid
    except ImportError:
        raise ImportError("Run:  pip install torchreid")

    print("  Initialising OSNet x0.25 (auto-download on first run)...")
    _OSNET = torchreid.models.build_model(
        name="osnet_x0_25", num_classes=1000, pretrained=True
    )
    _OSNET.eval().to(DEVICE)
    print("  OSNet ready.")
    return _OSNET


def extract_reid(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return L2-normalised 512-dim OSNet feature, or zeros on bad crop."""
    x1,y1,x2,y2 = (max(0,int(v)) for v in bbox)
    x2 = min(img.shape[1]-1, x2);  y2 = min(img.shape[0]-1, y2)
    if x2-x1 < 8 or y2-y1 < 8:
        return np.zeros(512, dtype=np.float32)
    crop = cv2.resize(img[y1:y2, x1:x2], (128, 256))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = (crop.astype(np.float32)/255.0 - _OSNET_MEAN) / _OSNET_STD
    t    = torch.FloatTensor(crop).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = _get_osnet()(t)[0].cpu().numpy().astype(np.float32)
    n = np.linalg.norm(f);  return f/(n+1e-8)


def reid_dist(a, b) -> float:
    if a is None or b is None: return 0.5
    return float((1.0 - np.dot(a, b)) / 2.0)


# ============================================================================
# MOTION PREDICTOR  (per-track)
# ============================================================================

class MotionPredictor:
    def __init__(self, model: nn.Module, norm_stats: NormStats):
        self.model      = model
        self.norm_stats = norm_stats
        self._hist: List[np.ndarray] = []
        self._smooth: Optional[np.ndarray] = None

    def update(self, bbox: np.ndarray):
        self._hist.append(bbox.copy())
        if len(self._hist) > SEQ_LEN:
            self._hist = self._hist[-SEQ_LEN:]
        self._smooth = None     # reset smoothing on real observation

    def predict(self) -> np.ndarray:
        raw = self._raw_predict()
        if self._smooth is None:
            self._smooth = raw
        else:
            self._smooth = EMA_ALPHA * raw + (1.0-EMA_ALPHA) * self._smooth
        return self._smooth.copy()

    def _raw_predict(self) -> np.ndarray:
        if not self._hist:
            return np.zeros(4, dtype=np.float32)
        if len(self._hist) < SEQ_LEN:
            return self._hist[-1].copy()

        feats  = _to_vel_features(self._hist[-SEQ_LEN:])
        normed = self.norm_stats.normalize(np.array(feats, dtype=np.float32))
        inp    = torch.FloatTensor(normed).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out_norm = self.model(inp)[0, 0].cpu().numpy()   # (4,)

        # Denormalise — pad vx,vy with zeros, take first 4 dims
        padded = np.concatenate([out_norm, np.zeros(2, dtype=np.float32)])
        cx,cy,w,h = self.norm_stats.denormalize(padded)[:4]
        w = max(w, 1.0);  h = max(h, 1.0)
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)


# ============================================================================
# TRACKER
# ============================================================================

class NeuralTracker:
    def __init__(self, model: nn.Module, norm_stats: NormStats):
        self.model        = model
        self.norm_stats   = norm_stats
        self.tracks: List[Track] = []
        self.next_id      = 1
        self._lost_buf: List[Track] = []    # for long-gap re-ID

    # ── public ──────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection],
               frame: Optional[np.ndarray] = None) -> List[Track]:

        # Extract ReID for all detections
        if frame is not None:
            for d in detections:
                d.reid_feat = extract_reid(frame, d.bbox)

        # Predict all active tracks
        self._predict_all()

        high = [d for d in detections if d.confidence >= 0.5]
        low  = [d for d in detections if d.confidence <  0.5]
        all_idx = list(range(len(self.tracks)))

        # Stage 1: high-conf ↔ active tracks
        m1, unm_high, unm_trk1 = self._match(high, all_idx, IOU_THRESH_HIGH)

        # Stage 2: low-conf ↔ remaining unmatched tracks
        m2, _, unm_trk2        = self._match(low,  unm_trk1, IOU_THRESH_LOW)

        matched_idx = set()
        for di, ti in m1:
            self._apply(self.tracks[ti], high[di], frame);  matched_idx.add(ti)
        for di, ti in m2:
            self._apply(self.tracks[ti], low[di],  frame);  matched_idx.add(ti)

        # Age unmatched tracks
        for ti in range(len(self.tracks)):
            if ti not in matched_idx:
                self.tracks[ti].time_since_update += 1
                self.tracks[ti].is_predicted = True

        # Move dying tracks to lost buffer
        alive, dying = [], []
        for t in self.tracks:
            (dying if t.time_since_update >= MAX_AGE else alive).append(t)
        self._lost_buf.extend([t for t in dying if t.hits >= MIN_HITS
                                and t.reid_feat is not None])
        self._lost_buf = self._lost_buf[-60:]
        self.tracks    = alive

        # New track or re-ID recovery for unmatched high-conf detections
        used_high = {di for di, _ in m1}
        for di, det in enumerate(high):
            if di in used_high:
                continue
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

    # ── private ─────────────────────────────────────────────────────────────

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
                cost[di, j] = ((1-REID_WEIGHT) * (1-iou)
                               + REID_WEIGHT   * reid_dist(det.reid_feat,
                                                            self.tracks[ti].reid_feat))

        ri, ci = linear_sum_assignment(cost)
        matched, used_d, used_t = [], set(), set()
        for r, c in zip(ri, ci):
            ti = trk_indices[c]
            if self._iou(dets[r].bbox, self.tracks[ti].bbox) >= iou_thr:
                matched.append((r, ti));  used_d.add(r);  used_t.add(ti)

        return (matched,
                [i  for i  in range(len(dets))     if i  not in used_d],
                [ti for ti in trk_indices            if ti not in used_t])

    def _reidentify(self, det: Detection) -> Optional[Track]:
        if det.reid_feat is None or not self._lost_buf:
            return None
        best_d, best_t = REID_LONG_THRESH, None
        for t in self._lost_buf:
            d = reid_dist(det.reid_feat, t.reid_feat)
            if d < best_d:
                best_d, best_t = d, t
        return best_t

    def _apply(self, trk: Track, det: Detection, frame):
        trk.bbox = det.bbox.copy()
        trk.confidence = det.confidence
        trk.trajectory.append(det.bbox.copy())
        trk.hits += 1
        trk.time_since_update = 0
        trk.is_predicted = False
        if trk._mp is None:
            trk._mp = MotionPredictor(self.model, self.norm_stats)
        trk._mp.update(det.bbox)
        if det.reid_feat is not None:
            if trk.reid_feat is None:
                trk.reid_feat = det.reid_feat.copy()
            else:
                trk.reid_feat = 0.8*trk.reid_feat + 0.2*det.reid_feat
                n = np.linalg.norm(trk.reid_feat)
                trk.reid_feat /= (n+1e-8)

    def _new_track(self, det: Detection):
        t = Track(id=self.next_id, bbox=det.bbox.copy(),
                  class_id=det.class_id, confidence=det.confidence,
                  trajectory=[det.bbox.copy()], hits=1,
                  reid_feat=det.reid_feat)
        t._mp = MotionPredictor(self.model, self.norm_stats)
        t._mp.update(det.bbox)
        self.tracks.append(t)
        self.next_id += 1

    @staticmethod
    def _iou(a, b) -> float:
        xi1=max(a[0],b[0]); yi1=max(a[1],b[1])
        xi2=min(a[2],b[2]); yi2=min(a[3],b[3])
        inter=max(0.,xi2-xi1)*max(0.,yi2-yi1)
        u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return float(inter/u) if u>0 else 0.


# ============================================================================
# DRAWING
# ============================================================================

_PAL = [(50,220,50),(80,160,255),(255,180,60),(180,80,255),
        (60,220,220),(255,80,160),(140,255,140),(255,140,80)]

def _col(tid): return _PAL[tid % len(_PAL)]

def _dashed_rect(img, p1, p2, color, thick=2, dash=10):
    x1,y1=p1; x2,y2=p2
    for (sx,sy,ex,ey) in [(x1,y1,x2,y1),(x2,y1,x2,y2),
                          (x2,y2,x1,y2),(x1,y2,x1,y1)]:
        dx,dy=ex-sx,ey-sy
        L=max(1,int(np.hypot(dx,dy))); steps=L//(dash*2)+1
        for s in range(steps):
            t0=min(1.,s*2*dash/L); t1=min(1.,(s*2+1)*dash/L)
            cv2.line(img,(int(sx+t0*dx),int(sy+t0*dy)),
                         (int(sx+t1*dx),int(sy+t1*dy)),color,thick)

def _lbl(img, txt, x, y, col):
    f=cv2.FONT_HERSHEY_SIMPLEX; sc,th=0.5,1
    (tw,fh),bl=cv2.getTextSize(txt,f,sc,th)
    cv2.rectangle(img,(x,max(0,y-fh-4)),(x+tw+6,y+bl),CLR_BLACK,-1)
    cv2.putText(img,txt,(x+3,y),f,sc,CLR_WHITE,th)

def draw_frame(img: np.ndarray, tracks: List[Track],
               video_id: str = "", is_video_frame: bool = False) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]

    for t in tracks:
        x1=max(0,int(t.bbox[0]));  y1=max(0,int(t.bbox[1]))
        x2=min(W-1,int(t.bbox[2]));y2=min(H-1,int(t.bbox[3]))
        if x2<=x1 or y2<=y1: continue

        if t.is_predicted:
            _dashed_rect(out,(x1,y1),(x2,y2),CLR_PREDICTED,thick=2)
            _lbl(out,f"ID:{t.id} OCCLUDED",x1,y1,CLR_PREDICTED)
        else:
            cv2.rectangle(out,(x1,y1),(x2,y2),_col(t.id),2)
            _lbl(out,f"ID:{t.id}  {t.confidence:.2f}",x1,y1,_col(t.id))

    # Watermark for the 3 output videos
    if is_video_frame and video_id:
        wm   = f"{video_id}_LSTM_Improved"
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc, th = 0.65, 2
        (tw,fh),bl = cv2.getTextSize(wm,font,sc,th)
        wx, wy = 10, H-10
        cv2.rectangle(out,(wx-4,wy-fh-6),(wx+tw+4,wy+bl+2),CLR_BLACK,-1)
        cv2.putText(out,wm,(wx,wy),font,sc,CLR_WHITE,th)

    return out


# ============================================================================
# EVALUATION
# ============================================================================

def compute_ade(pt_list, gt_list) -> float:
    errors = []
    for pt in pt_list:
        if not pt: continue
        best = float("inf")
        for gt in gt_list:
            if not gt: continue
            n  = min(len(pt),len(gt))
            pc = np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in pt[:n]])
            gc = np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in gt[:n]])
            best = min(best, float(np.mean(np.linalg.norm(pc-gc,axis=1))))
        if best < float("inf"): errors.append(best)
    return float(np.mean(errors)) if errors else 0.0


def evaluate_tracking(predictions, annotations, vname):
    vinfo = None
    for v in annotations["videos"]:
        if v["file_names"][0].replace("\\","/").split("/")[0] == vname:
            vinfo = v; break
    if vinfo is None: return None

    vid_id  = vinfo["id"]
    vid_len = vinfo.get("length", len(vinfo.get("file_names",[])))

    gt_by_frame = defaultdict(list)
    for ann in annotations["annotations"]:
        if ann["video_id"] != vid_id: continue
        tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        for fi, bbox in enumerate(ann.get("bboxes",[])):
            if not bbox or len(bbox)!=4: continue
            x,y,w,h = bbox
            if w>0 and h>0:
                gt_by_frame[fi].append({"id":tid,"bbox":[x,y,x+w,y+h]})

    acc   = mm.MOTAccumulator(auto_id=True)
    pt    = defaultdict(list)
    gt    = defaultdict(list)

    for pf in predictions:
        fid = pf["frame_id"]
        if fid >= vid_len: continue
        gts   = gt_by_frame.get(fid,[])
        g_ids = [o["id"]   for o in gts]
        g_bb  = [o["bbox"] for o in gts]
        p_ids = [t[0] for t in pf["tracks"]]
        p_bb  = [t[1] for t in pf["tracks"]]
        for tid,bb in zip(p_ids,p_bb): pt[tid].append(bb)
        for tid,bb in zip(g_ids,g_bb): gt[tid].append(bb)
        dist = (np.array([[1-NeuralTracker._iou(np.array(g),np.array(p))
                           for p in p_bb] for g in g_bb])
                if g_bb and p_bb else np.empty((len(g_bb),len(p_bb))))
        acc.update(g_ids,p_ids,dist)

    mh  = mm.metrics.create()
    s   = mh.compute(acc,metrics=["mota","num_switches","idf1"],name="acc")
    return {
        "mota":        float(s["mota"].values[0])         if "mota"         in s.columns else 0.,
        "idf1":        float(s["idf1"].values[0])         if "idf1"         in s.columns else 0.,
        "id_switches": int(s["num_switches"].values[0])   if "num_switches" in s.columns else 0,
        "ade":         compute_ade(list(pt.values()),list(gt.values())),
    }


def aggregate_and_normalize(ml):
    a_mota = float(np.mean([m["mota"] for m in ml]))
    a_idf1 = float(np.mean([m["idf1"] for m in ml]))
    a_ade  = float(np.mean([m["ade"]  for m in ml]))
    sw     = int(np.sum( [m["id_switches"] for m in ml]))
    return {
        "mota_norm":   round(max(0.,min(1.,a_mota)),       4),
        "idf1_norm":   round(max(0.,min(1.,a_idf1)),       4),
        "ade_norm":    round(1.-min(1.,a_ade/200.),        4),
        "avg_ade_px":  round(a_ade, 2),
        "id_switches": sw,
    }


# ============================================================================
# TRAJECTORY COLLECTION
# ============================================================================

def collect_trajectories(folders, annotations):
    trajs = []
    for i, vf in enumerate(folders, 1):
        vname = vf.name
        print(f"  [{i}/{len(folders)}] {vname}")
        vinfo = None
        for v in annotations["videos"]:
            if v["file_names"][0].replace("\\","/").split("/")[0] == vname:
                vinfo = v; break
        if vinfo is None: continue
        gt = defaultdict(list)
        for ann in annotations["annotations"]:
            if ann["video_id"] != vinfo["id"]: continue
            tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
            for bbox in ann.get("bboxes",[]):
                if not bbox or len(bbox)!=4: continue
                x,y,w,h = bbox
                if w>0 and h>0: gt[tid].append([x,y,x+w,y+h])
        for traj in gt.values():
            if len(traj) >= SEQ_LEN + PRED_LEN + 1:
                trajs.append(traj)
    return trajs


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: nn.Module, loader: DataLoader) -> nn.Module:
    model     = model.to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        model.train();  total = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            pred = model(inp)                       # (B, 1, 4)
            loss = criterion(pred, tgt[:,:,:4])     # supervise on cx,cy,w,h
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"    Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Loss: {total/len(loader):.6f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
    return model


# ============================================================================
# TEST LOOP  —  metrics + video for specific IDs
# ============================================================================

def run_test(model, norm_stats, test_folders, annotations, detector):
    """
    Iterates all test_folders for metrics.
    For folders whose name is in VIDEO_IDS, also writes an annotated MP4
    to the refined_model/ directory with a watermark.
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_metrics = []

    for idx, vf in enumerate(test_folders, 1):
        vname      = vf.name
        frames     = sorted(vf.glob("*.jpg"))
        is_vid     = vname in VIDEO_IDS
        print(f"  [{idx}/{len(test_folders)}] {vname}"
              f"  {'← video' if is_vid else ''}")
        if not frames: continue

        # Set up writer only for the 3 target videos
        writer = None
        if is_vid:
            sample = cv2.imread(str(frames[0]))
            if sample is not None:
                H, W = sample.shape[:2]
                out_path = VIDEO_DIR / f"{vname}_LSTM_Improved.mp4"
                writer   = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS_OUT, (W, H),
                )

        tracker     = NeuralTracker(model, norm_stats)
        predictions = []

        for fi, fpath in enumerate(frames):
            img = cv2.imread(str(fpath))
            if img is None: continue

            dets = []
            for res in detector(img, conf=0.4, verbose=False):
                for i in range(len(res.boxes)):
                    dets.append(Detection(
                        bbox=res.boxes.xyxy[i].cpu().numpy().astype(np.float32),
                        confidence=float(res.boxes.conf[i]),
                        class_id=int(res.boxes.cls[i]),
                        frame_id=fi,
                    ))

            tracks = tracker.update(dets, frame=img)
            predictions.append({
                "frame_id": fi,
                "tracks":   [(t.id, t.bbox.tolist(), t.class_id, t.confidence)
                             for t in tracks],
            })

            if writer is not None:
                writer.write(draw_frame(img, tracks,
                                        video_id=vname, is_video_frame=True))

        if writer is not None:
            writer.release()
            print(f"    ✔  Video → {VIDEO_DIR}/{vname}_LSTM_Improved.mp4")

        m = evaluate_tracking(predictions, annotations, vname)
        if m: video_metrics.append(m)

    return video_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    import os
    print(f"Working directory: {os.getcwd()}")

    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 62)
    print("  LSTM Improved Tracker")
    print(f"  Device : {DEVICE}")
    print("=" * 62)

    # Load annotations
    with open(ANNOT_FILE) as f:
        annotations = json.load(f)

    all_folders   = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    train_folders = all_folders[TRAIN_START : TRAIN_START + TRAIN_COUNT]
    test_folders  = all_folders[TEST_START  : TEST_START  + TEST_COUNT]

    print(f"Train : {len(train_folders)} videos  "
          f"(index {TRAIN_START}–{TRAIN_START+TRAIN_COUNT-1})")
    print(f"Test  : {len(test_folders)} videos  "
          f"(index {TEST_START}–{TEST_START+TEST_COUNT-1})")
    print(f"Videos: {VIDEO_IDS}\n")

    # ── Collect trajectories ────────────────────────────────────────────────
    print("Collecting training trajectories...")
    trajs = collect_trajectories(train_folders, annotations)
    print(f"Total: {len(trajs)} trajectories\n")

    # ── Normalisation ───────────────────────────────────────────────────────
    norm_stats = NormStats()
    norm_stats.fit(trajs)
    norm_stats.save(MODELS_DIR / "norm_stats_improved.npz")

    # ── Dataset / loader ────────────────────────────────────────────────────
    dataset = TrajectoryDataset(trajs, norm_stats)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=(DEVICE=="cuda"))
    print(f"Training samples : {len(dataset)}\n")

    # ── Load pretrained model weights (training disabled) ───────────────────
    print("Skipping training – loading pretrained LSTM weights instead")
    model = LSTMImproved()
    model = load_model_weights(model, MODELS_DIR / "lstm_improved.pth")
    # if weights file doesn't exist, running this will raise an error
    # you may want to train and save separately before re-running


    # ── Initialise OSNet once (triggers download if needed) ─────────────────
    print("\nInitialising OSNet ReID...")
    _get_osnet()

    # ── Detector ────────────────────────────────────────────────────────────
    detector = YOLO("yolo11n.pt")

    # ── Test + produce videos ───────────────────────────────────────────────
    print(f"\nTesting on {TEST_COUNT} videos "
          f"(producing MP4 for {VIDEO_IDS})...")
    video_metrics = run_test(model, norm_stats, test_folders,
                             annotations, detector)

    # ── Metrics ─────────────────────────────────────────────────────────────
    if video_metrics:
        m = aggregate_and_normalize(video_metrics)

        print(f"\n{'='*62}")
        print("  RESULTS  —  LSTM_Improved  (normalised, higher = better)")
        print(f"{'='*62}")
        print(f"  MOTA (norm)  : {m['mota_norm']}")
        print(f"  IDF1 (norm)  : {m['idf1_norm']}")
        print(f"  ADE  (norm)  : {m['ade_norm']}   "
              f"(raw ADE: {m['avg_ade_px']} px)")
        print(f"  ID Switches  : {m['id_switches']}")
        print(f"{'='*62}")

        df = pd.DataFrame([{"Model": "LSTM_Improved", **m}])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n  CSV  → {OUTPUT_CSV}")

    # ── File summary ────────────────────────────────────────────────────────
    print(f"\n  Models  → {MODELS_DIR.resolve()}")
    print(f"  Videos  → {VIDEO_DIR.resolve()}")
    for p in sorted(VIDEO_DIR.glob("*.mp4")):
        sz = p.stat().st_size / 1_048_576
        print(f"    {p.name}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()