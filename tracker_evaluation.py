"""
OVIS LSTM and Transformer Tracking Evaluation
Tests LSTM and Transformer models with YOLOv11n for motion prediction
Training: Videos 51-350 (300 videos)
Testing: Videos 1-50 (50 videos)
Epochs: 15 for each model

Saved files (in MODELS_DIR = ./saved_models/):
  lstm_model.pth          - LSTM weights
  transformer_model.pth   - Transformer weights
  norm_stats.npz          - Normalization mean/std (shared between both models)
"""

import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List
from dataclasses import dataclass
from collections import defaultdict
from ultralytics import YOLO
import motmetrics as mm


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    class_id: int
    frame_id: int = 0


@dataclass
class Track:
    id: int
    bbox: np.ndarray
    class_id: int
    confidence: float
    trajectory: List[np.ndarray]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


# ============================================================================
# NORMALIZATION STATS
# ============================================================================

class NormStats:
    """Per-dimension mean/std normalization for [cx, cy, w, h]."""

    def __init__(self):
        self.mean = np.zeros(4, dtype=np.float32)
        self.std  = np.ones(4,  dtype=np.float32)

    def fit(self, trajectories: List[List]):
        all_feats = []
        for traj in trajectories:
            for bbox in traj:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w  =  bbox[2] - bbox[0]
                h  =  bbox[3] - bbox[1]
                all_feats.append([cx, cy, w, h])
        arr = np.array(all_feats, dtype=np.float32)
        self.mean = arr.mean(axis=0)
        self.std  = arr.std(axis=0) + 1e-8

    def normalize(self, feat: np.ndarray) -> np.ndarray:
        return (feat - self.mean) / self.std

    def denormalize(self, feat: np.ndarray) -> np.ndarray:
        return feat * self.std + self.mean

    def save(self, path: Path):
        np.savez(path, mean=self.mean, std=self.std)
        print(f"  Saved norm stats  -> {path}")

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        data = np.load(path)
        ns = cls()
        ns.mean = data["mean"]
        ns.std  = data["std"]
        print(f"  Loaded norm stats <- {path}")
        return ns


# ============================================================================
# TRAJECTORY DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, norm_stats: NormStats, seq_len=10, pred_len=1):
        self.norm_stats = norm_stats
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.samples    = []

        for traj in sequences:
            if len(traj) < seq_len + pred_len:
                continue
            for i in range(len(traj) - seq_len - pred_len + 1):
                inp = self._to_features(traj[i : i + seq_len])
                tgt = self._to_features(traj[i + seq_len : i + seq_len + pred_len])
                self.samples.append({
                    "input":  norm_stats.normalize(inp),
                    "target": norm_stats.normalize(tgt),
                })

    def _to_features(self, bboxes) -> np.ndarray:
        feats = []
        for bbox in bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w  =  bbox[2] - bbox[0]
            h  =  bbox[3] - bbox[1]
            feats.append([cx, cy, w, h])
        return np.array(feats, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.FloatTensor(s["input"]), torch.FloatTensor(s["target"])


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


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerPredictor(nn.Module):
    def __init__(self, input_size=4, d_model=64, nhead=4, num_layers=2, pred_len=1):
        super().__init__()
        self.pred_len    = pred_len
        self.input_proj  = nn.Linear(input_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc          = nn.Linear(d_model, input_size * pred_len)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).view(-1, self.pred_len, 4)


# ============================================================================
# MODEL SAVE / LOAD
# ============================================================================

def save_model(model: nn.Module, path: Path):
    torch.save(model.state_dict(), path)
    print(f"  Saved model       -> {path}")


def load_model(model: nn.Module, path: Path, device: str) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded model      <- {path}")
    return model


# ============================================================================
# MOTION PREDICTOR WRAPPER
# ============================================================================

class MotionPredictor:
    """
    Per-track wrapper around a trained model.
    Maintains a rolling history window, normalizes inputs, denormalizes outputs.
    Falls back to the last known bbox when history is too short.
    """
    def __init__(self, model: nn.Module, norm_stats: NormStats,
                 device: str, seq_len: int = 10):
        self.model      = model
        self.norm_stats = norm_stats
        self.device     = device
        self.seq_len    = seq_len
        self.trajectory: List[np.ndarray] = []

    def update(self, bbox: np.ndarray):
        self.trajectory.append(bbox.copy())
        if len(self.trajectory) > self.seq_len:
            self.trajectory = self.trajectory[-self.seq_len:]

    def predict(self) -> np.ndarray:
        if len(self.trajectory) < self.seq_len:
            return self.trajectory[-1].copy() if self.trajectory else np.zeros(4)

        feats = []
        for bbox in self.trajectory[-self.seq_len:]:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w  =  bbox[2] - bbox[0]
            h  =  bbox[3] - bbox[1]
            feats.append([cx, cy, w, h])

        feats_norm = self.norm_stats.normalize(np.array(feats, dtype=np.float32))
        inp = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(inp)[0, 0].cpu().numpy()

        cx, cy, w, h = self.norm_stats.denormalize(pred_norm)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


# ============================================================================
# TRACKER
# ============================================================================

class NeuralTracker:
    """
    IoU-based multi-object tracker with neural motion prediction.
    During occlusion the model predicts the object's position for up to
    `max_age` frames before the track is dropped.
    """
    def __init__(self, model: nn.Module, norm_stats: NormStats,
                 device: str, max_age: int = 30,
                 min_hits: int = 3, iou_threshold: float = 0.3):
        self.model         = model
        self.norm_stats    = norm_stats
        self.device        = device
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 1

    def _make_predictor(self) -> MotionPredictor:
        return MotionPredictor(self.model, self.norm_stats, self.device)

    def update(self, detections: List[Detection]) -> List[Track]:
        # 1. Predict next position for every active track
        for track in self.tracks:
            if not hasattr(track, "predictor"):
                track.predictor = self._make_predictor()
                for bbox in track.trajectory:
                    track.predictor.update(bbox)
            track.bbox = track.predictor.predict()

        # 2. Match detections → tracks via IoU
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # 3. Update matched tracks with real detections
        for det_idx, trk_idx in matched:
            trk = self.tracks[trk_idx]
            det = detections[det_idx]
            trk.bbox = det.bbox
            trk.confidence = det.confidence
            trk.trajectory.append(det.bbox.copy())
            trk.hits += 1
            trk.time_since_update = 0
            trk.predictor.update(det.bbox)

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det   = detections[det_idx]
            track = Track(
                id=self.next_id, bbox=det.bbox, class_id=det.class_id,
                confidence=det.confidence, trajectory=[det.bbox.copy()], hits=1,
            )
            track.predictor = self._make_predictor()
            track.predictor.update(det.bbox)
            self.tracks.append(track)
            self.next_id += 1

        # 5. Age unmatched (occluded) tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].time_since_update += 1

        # 6. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # 7. Return only confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _match(self, detections):
        if not self.tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.tracks)))

        iou_mat = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                iou_mat[d, t] = self._iou(det.bbox, trk.bbox)

        matched = []
        while iou_mat.max() > self.iou_threshold:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched.append((i, j))
            iou_mat[i, :] = 0
            iou_mat[:, j] = 0

        matched_d = {m[0] for m in matched}
        matched_t = {m[1] for m in matched}
        return (matched,
                [d for d in range(len(detections)) if d not in matched_d],
                [t for t in range(len(self.tracks))  if t not in matched_t])

    @staticmethod
    def _iou(b1, b2):
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def compute_ade(predicted_trajectories, gt_trajectories):
    if not predicted_trajectories or not gt_trajectories:
        return 0.0
    errors = []
    for pred_traj in predicted_trajectories:
        if not pred_traj:
            continue
        best = float("inf")
        for gt_traj in gt_trajectories:
            if not gt_traj:
                continue
            n  = min(len(pred_traj), len(gt_traj))
            pc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in pred_traj[:n]])
            gc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in gt_traj[:n]])
            err = np.mean(np.linalg.norm(pc - gc, axis=1))
            best = min(best, err)
        if best != float("inf"):
            errors.append(best)
    return float(np.mean(errors)) if errors else 0.0


def evaluate_tracking(predictions, annotations, video_folder_name):
    video_info = None
    for video in annotations["videos"]:
        folder = video["file_names"][0].replace("\\", "/").split("/")[0]
        if folder == video_folder_name:
            video_info = video
            break
    if video_info is None:
        return None

    video_id     = video_info["id"]
    video_length = video_info.get("length", len(video_info.get("file_names", [])))

    gt_by_frame = defaultdict(list)
    for ann in annotations["annotations"]:
        if ann["video_id"] != video_id:
            continue
        tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
        for frame_idx, bbox in enumerate(ann.get("bboxes", [])):
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w > 0 and h > 0:
                gt_by_frame[frame_idx].append({"id": tid, "bbox": [x, y, x+w, y+h]})

    acc        = mm.MOTAccumulator(auto_id=True)
    pred_trajs = defaultdict(list)
    gt_trajs   = defaultdict(list)

    for pf in predictions:
        frame_id = pf["frame_id"]
        if frame_id >= video_length:
            continue
        gt_objs     = gt_by_frame.get(frame_id, [])
        gt_ids      = [o["id"]   for o in gt_objs]
        gt_bboxes   = [o["bbox"] for o in gt_objs]
        pred_ids    = [t[0] for t in pf["tracks"]]
        pred_bboxes = [t[1] for t in pf["tracks"]]

        for tid, bbox in zip(pred_ids, pred_bboxes):
            pred_trajs[tid].append(bbox)
        for tid, bbox in zip(gt_ids, gt_bboxes):
            gt_trajs[tid].append(bbox)

        if gt_bboxes and pred_bboxes:
            dist = np.array([
                [1 - NeuralTracker._iou(np.array(g), np.array(p)) for p in pred_bboxes]
                for g in gt_bboxes
            ])
        else:
            dist = np.empty((len(gt_bboxes), len(pred_bboxes)))

        acc.update(gt_ids, pred_ids, dist)

    mh      = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "num_switches", "idf1"], name="acc")
    ade     = compute_ade(list(pred_trajs.values()), list(gt_trajs.values()))

    return {
        "mota":        float(summary["mota"].values[0])           if "mota"         in summary.columns else 0.0,
        "idf1":        float(summary["idf1"].values[0])           if "idf1"         in summary.columns else 0.0,
        "id_switches": int(summary["num_switches"].values[0])     if "num_switches" in summary.columns else 0,
        "ade":         ade,
    }


def aggregate_and_normalize(video_metrics):
    avg_mota = float(np.mean([m["mota"] for m in video_metrics]))
    avg_idf1 = float(np.mean([m["idf1"] for m in video_metrics]))
    avg_ade  = float(np.mean([m["ade"]  for m in video_metrics]))
    total_sw = int(np.sum( [m["id_switches"] for m in video_metrics]))

    # Clamp MOTA/IDF1 to [0, 1]; invert ADE (higher score = lower pixel error)
    mota_norm = max(0.0, min(1.0, avg_mota))
    idf1_norm = max(0.0, min(1.0, avg_idf1))
    ade_norm  = 1.0 - min(1.0, avg_ade / 200.0)

    return {
        "mota_norm":   round(mota_norm, 4),
        "idf1_norm":   round(idf1_norm, 4),
        "ade_norm":    round(ade_norm,  4),
        "avg_ade_px":  round(avg_ade,   2),
        "id_switches": total_sw,
    }


# ============================================================================
# TRAJECTORY COLLECTION
# ============================================================================

def collect_trajectories(video_folders, annotations):
    trajectories = []
    for idx, vf in enumerate(video_folders, 1):
        vname = vf.name
        print(f"  Collecting [{idx}/{len(video_folders)}] {vname}")

        video_info = None
        for video in annotations["videos"]:
            folder = video["file_names"][0].replace("\\", "/").split("/")[0]
            if folder == vname:
                video_info = video
                break
        if video_info is None:
            continue

        gt_trajs = defaultdict(list)
        for ann in annotations["annotations"]:
            if ann["video_id"] != video_info["id"]:
                continue
            tid = ann.get("instance_id") or ann.get("track_id") or ann["id"]
            for bbox in ann.get("bboxes", []):
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    gt_trajs[tid].append([x, y, x+w, y+h])

        for traj in gt_trajs.values():
            if len(traj) >= 11:
                trajectories.append(traj)

    return trajectories


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: nn.Module, loader: DataLoader,
                epochs: int = 15, lr: float = 1e-3,
                device: str = "cpu") -> nn.Module:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch [{epoch}/{epochs}]  Loss: {total_loss / len(loader):.6f}")

    return model


# ============================================================================
# TEST LOOP (shared between both models)
# ============================================================================

def test_model(model: nn.Module, norm_stats: NormStats,
               test_folders, annotations, detector: YOLO,
               device: str, model_name: str) -> List[dict]:
    print(f"\nTesting {model_name}...")
    video_metrics = []

    for idx, video_folder in enumerate(test_folders, 1):
        vname  = video_folder.name
        frames = sorted(video_folder.glob("*.jpg"))
        print(f"  [{idx}/{len(test_folders)}] {vname}  ({len(frames)} frames)")
        if not frames:
            continue

        tracker     = NeuralTracker(model, norm_stats, device=device)
        predictions = []

        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            detections = []
            for result in detector(img, conf=0.5, verbose=False):
                boxes = result.boxes
                for i in range(len(boxes)):
                    detections.append(Detection(
                        bbox=boxes.xyxy[i].cpu().numpy(),
                        confidence=float(boxes.conf[i].cpu().numpy()),
                        class_id=int(boxes.cls[i].cpu().numpy()),
                        frame_id=frame_idx,
                    ))

            tracks = tracker.update(detections)
            predictions.append({
                "frame_id": frame_idx,
                "tracks":   [(t.id, t.bbox.tolist(), t.class_id, t.confidence)
                             for t in tracks],
            })

        metrics = evaluate_tracking(predictions, annotations, vname)
        if metrics:
            video_metrics.append(metrics)

    return video_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    TRAIN_DIR         = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\train")
    TRAIN_ANNOTATIONS = Path(r"C:\Users\mayoo\OneDrive - University of Jeddah\Ai 5th term\Graduation Project\Trying Baseline Models to evaluate\annotations_train.json")
    MODELS_DIR        = Path("saved_models")
    OUTPUT_CSV        = Path("tracking_results_neural.csv")

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Config ─────────────────────────────────────────────────────────────
    TRAIN_START = 50
    TRAIN_COUNT = 300
    TEST_START  = 0
    TEST_COUNT  = 50
    EPOCHS      = 15
    BATCH_SIZE  = 32
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")

    # ── Load annotations ───────────────────────────────────────────────────
    with open(TRAIN_ANNOTATIONS) as f:
        annotations = json.load(f)

    all_folders   = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    train_folders = all_folders[TRAIN_START : TRAIN_START + TRAIN_COUNT]
    test_folders  = all_folders[TEST_START  : TEST_START  + TEST_COUNT]

    print(f"Train videos : {len(train_folders)}  (index {TRAIN_START}–{TRAIN_START+TRAIN_COUNT-1})")
    print(f"Test  videos : {len(test_folders)}   (index {TEST_START}–{TEST_START+TEST_COUNT-1})")

    # ── Collect training trajectories ──────────────────────────────────────
    print("\nCollecting training trajectories...")
    train_trajectories = collect_trajectories(train_folders, annotations)
    print(f"Total training trajectories: {len(train_trajectories)}")

    # ── Fit & save normalization stats ─────────────────────────────────────
    norm_stats = NormStats()
    norm_stats.fit(train_trajectories)
    norm_stats.save(MODELS_DIR / "norm_stats.npz")

    # ── Build dataset / dataloader ─────────────────────────────────────────
    dataset = TrajectoryDataset(train_trajectories, norm_stats, seq_len=10, pred_len=1)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Training samples: {len(dataset)}")

    detector = YOLO("yolo11n.pt")
    results  = []

    # ════════════════════════════════════════════════════════════════════════
    # LSTM
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("TRAINING LSTM")
    print("="*70)

    lstm_model = LSTMPredictor(input_size=4, hidden_size=64, num_layers=2, pred_len=1)
    lstm_model = train_model(lstm_model, loader, epochs=EPOCHS, device=DEVICE)
    save_model(lstm_model, MODELS_DIR / "lstm_model.pth")

    lstm_metrics = test_model(lstm_model, norm_stats, test_folders,
                              annotations, detector, DEVICE, "LSTM")
    if lstm_metrics:
        m = aggregate_and_normalize(lstm_metrics)
        results.append({"Model": "LSTM", **m})
        print(f"\nLSTM Results:")
        print(f"  MOTA (norm): {m['mota_norm']}   IDF1 (norm): {m['idf1_norm']}")
        print(f"  ADE  (norm): {m['ade_norm']}    ADE (px):    {m['avg_ade_px']}")
        print(f"  ID Switches: {m['id_switches']}")

    # ════════════════════════════════════════════════════════════════════════
    # TRANSFORMER
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("TRAINING TRANSFORMER")
    print("="*70)

    trans_model = TransformerPredictor(input_size=4, d_model=64, nhead=4,
                                       num_layers=2, pred_len=1)
    trans_model = train_model(trans_model, loader, epochs=EPOCHS, device=DEVICE)
    save_model(trans_model, MODELS_DIR / "transformer_model.pth")

    trans_metrics = test_model(trans_model, norm_stats, test_folders,
                               annotations, detector, DEVICE, "Transformer")
    if trans_metrics:
        m = aggregate_and_normalize(trans_metrics)
        results.append({"Model": "Transformer", **m})
        print(f"\nTransformer Results:")
        print(f"  MOTA (norm): {m['mota_norm']}   IDF1 (norm): {m['idf1_norm']}")
        print(f"  ADE  (norm): {m['ade_norm']}    ADE (px):    {m['avg_ade_px']}")
        print(f"  ID Switches: {m['id_switches']}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved -> {OUTPUT_CSV}")
        print(df.to_string(index=False))

    print(f"\nAll model files saved in: {MODELS_DIR.resolve()}")
    print("  lstm_model.pth")
    print("  transformer_model.pth")
    print("  norm_stats.npz")


# ============================================================================
# HOW TO RELOAD SAVED MODELS (inference / future runs)
# ============================================================================
#
#   from ovis_tracking_eval import (
#       LSTMPredictor, TransformerPredictor, NormStats, load_model
#   )
#   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
#   norm_stats  = NormStats.load("saved_models/norm_stats.npz")
#
#   lstm_model  = load_model(LSTMPredictor(),        "saved_models/lstm_model.pth",        DEVICE)
#   trans_model = load_model(TransformerPredictor(), "saved_models/transformer_model.pth", DEVICE)
#
# ============================================================================

if __name__ == "__main__":
    main()