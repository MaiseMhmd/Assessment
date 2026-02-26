# Experiments

This repository contains two main experiments related to motion prediction for multi-object tracking.

---

## Experiment 1 — Baseline Tracker Evaluation

**File:** `tracker_evaluation.py`

This script evaluates tracking performance using YOLO detections combined with motion prediction models. The experiment:

- Collects trajectories from the dataset
- Trains motion prediction models
- Runs multi-object tracking
- Computes evaluation metrics such as MOTA, IDF1, ADE, and ID Switches
- Saves the results into a CSV file

---

## Experiment 2 — Improved LSTM Motion Model

**File:** `improving_lstm.py`

This experiment improves the baseline LSTM motion prediction model to better handle challenging scenarios such as occlusion and object reappearance. The script trains the improved LSTM model and evaluates its tracking performance compared to the baseline.

---

## Dataset

To reproduce the experiments, download the dataset from the following link:

[OVIS Training Data & Annotations](https://drive.google.com/drive/u/0/folders/1eE4lLKCbv54E866XBVce_ebh3oXYq99b)

From this dataset you will need:

- The training video folder
- The file `annotations_train.json`

Place them in your project directory as follows:

```
project/
│
├── train/
│   ├── video_001/
│   ├── video_002/
│   └── ...
│
└── annotations_train.json
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```
