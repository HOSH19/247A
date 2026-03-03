# LSTM Implementation Work Log

## Goal

Replace the TDSConvEncoder in the baseline pipeline with a BiLSTM encoder and systematically study the effect of architecture, preprocessing, and data characteristics on CER.

```
SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → [Encoder] → Linear → LogSoftmax → CTCLoss
```

MLP and CTC decoder kept identical to TDSConv baseline. Only the encoder and preprocessing are varied.

---

## Part 1: Architecture Experiments

Fixed: 16ch, 16 sessions, hop=16 (125Hz), standard augmentation.

### 150 Epoch Full Runs

| Model | Total Params | Val CER | Test CER | Epochs | Notes |
|-------|-------------|---------|----------|--------|-------|
| TDSConv (baseline) | 5.3M | 18.94 | 22.17 | 150 | logs/2026-02-27/16-29-13 |
| BiLSTM (h=384, l=2) | 8.2M | **14.55** | **15.76** | 150 | logs/2026-02-28/00-52-40 |
| TDS+BiLSTM hybrid | 13.0M | 14.58 | 15.47 | 150 | warm init ep38 |
| BiLSTM (h=512, l=3) | 19.1M | 15.91 | 22.80 | 150 | best ep135 |
| BiLSTM + Transformer | 22.3M | 14.67 | 17.25 | 150 | best ep129 |

### 40 Epoch Screening (Architecture / Scale)

| Model | Total Params | Val CER | Test CER | Epochs |
|-------|-------------|---------|----------|--------|
| TDSConv | 5.3M | 22.55 | 24.18 | 40 |
| BiLSTM (h=384, l=2) | 8.2M | 19.87 | 20.08 | 40 |
| BiLSTM (h=384, l=3) | 11.7M | 20.87 | 24.08 | 40 |
| BiLSTM (h=512, l=2) | 12.8M | 19.74 | 19.69 | 40 |
| BiLSTM (h=512, l=3) | 19.1M | **17.88** | 21.55 | 40 |
| BiLSTM + Transformer (screening) | 22.3M | 19.03 | 26.56 | 40 |

### Experiment Notes

**Exp 1 — BiLSTM vs TDSConv:**
TDSConv sees only a fixed 62ms local window. BiLSTM (14.55) outperforms TDSConv (18.94) by **−4.39 val CER**. Full-sequence bidirectional context is clearly beneficial.

**Exp 2 — TDS+BiLSTM Hybrid:**
Stacking TDSConv (local) before BiLSTM (global) gives no improvement (14.58 ≈ 14.55). BiLSTM already captures local patterns recurrently; TDS preprocessing is redundant.

**Exp 3 — Scale up BiLSTM:**
2×2 factorial (h×l) at 40 epochs: h=512, l=3 best at 40ep (17.88). Full 150ep run gives 15.91 — *worse* than h=384, l=2 (14.55). Larger model overfits on single-user data. Data, not capacity, is the bottleneck.

**Exp 4 — BiLSTM + Transformer:**
Self-attention on top of BiLSTM converges to same performance (14.67 ≈ 14.55) but slower. BiLSTM bidirectional states already capture sufficient context. Extra parameters don't help on small datasets.

---

## Part 2: Preprocessing Ablation

Fixed: BiLSTM h=384, l=2 (8.2M params), 16ch, 16 sessions.

### Sampling Rate (hop_length) — 40 Epochs

| hop_length | Effective Rate | Val CER | Test CER |
|-----------|----------------|---------|----------|
| 8 | 250 Hz | 26.47 | 25.14 |
| 16 | 125 Hz (baseline) | 19.87 | 20.08 |
| 24 | 83 Hz | 18.76 | 18.50 |
| 32 | 62.5 Hz | 17.50 | 16.99 |
| 40 | 50 Hz | 18.17 | 18.31 |
| **48** | **41.7 Hz** | **17.01** | **17.59** |
| 56 | 35.7 Hz | 17.92 | 18.22 |
| 64 | 31.25 Hz | 17.68 | 17.53 |

**hop=48 Full Run (150 epochs):**

| Model | Total Params | Val CER | Test CER | Epochs |
|-------|-------------|---------|----------|--------|
| BiLSTM hop=16 (baseline) | 8.2M | 14.55 | 15.76 | 150 |
| **BiLSTM hop=48** | **8.2M** | **13.98** | **14.52** | **150** |

**Insight:** Lower temporal resolution (hop=32~64) outperforms baseline (hop=16). Shorter sequences improve BiLSTM gradient flow, and EMG keystroke patterns (~50–200ms) don't require 125Hz resolution. hop=48 (41.7Hz) is the sweet spot.

### Data Augmentation — 40 Epochs (hop=16)

| Augmentation | Val CER | Test CER |
|---|---|---|
| Baseline (RandomBandRot + TemporalJitter + SpecAugment) | 19.87 | 20.08 |
| + GaussianNoise (std=0.1, post-spectrogram) | 20.16 | 20.34 |
| + AmplitudeScale (×0.7~1.3, pre-spectrogram) | 21.02 | 21.98 |

**Insight:** Both augmentations slightly hurt. The baseline already has three augmentation layers — adding more regularization slows convergence without generalization gain on this small dataset.

---

## Part 3: Data Ablation

Fixed: BiLSTM h=384, l=2 (8.2M params), hop=16, 40 epochs.

### Electrode Channels per Band

| Channels | in_features | Val CER | Test CER |
|----------|------------|---------|----------|
| 16 (full) | 528 | 19.87 | 20.08 |
| 8 | 264 | 25.88 | 26.35 |
| 4 | 132 | 36.53 | 37.97 |
| 2 | 66 | 66.59 | 67.80 |
| 1 | 33 | 88.04 | 86.90 |

**Insight:** Monotonic steep degradation as channels decrease. CER roughly doubles every halving. Each electrode captures spatially distinct, non-redundant muscle activation patterns — all 16 channels are necessary.

### Training Sessions

| Train sessions | Fraction | Val CER | Test CER |
|---|---|---|---|
| 2 | 12.5% | ~100 (fails) | ~100 |
| 4 | 25% | ~100 (fails) | ~100 |
| 8 | 50% | 36.97 | 33.46 |
| 16 (full) | 100% | **19.87** | **20.08** |

**Insight:** Sharp threshold between 4 and 8 sessions. Below 8, model immediately overfits (val CER ~100). All 16 sessions needed for competitive performance. Strongly confirms data-bottleneck hypothesis from architecture experiments.

---

## Implementation Details

### Key Files

| File | Description |
|------|-------------|
| `emg2qwerty/modules.py` | `LSTMEncoder`, `ChannelSlice` classes |
| `emg2qwerty/lightning.py` | `LSTMCTCModule`, `LSTMTransformerCTCModule` |
| `emg2qwerty/transforms.py` | `GaussianNoise`, `AmplitudeScale` classes added |
| `config/model/lstm_ctc.yaml` | BiLSTM config (h=384, l=2) |
| `config/model/lstm_transformer_ctc.yaml` | BiLSTM + Transformer config |
| `config/transforms/log_spectrogram_hop*.yaml` | Sampling rate variants (hop 8/16/24/32/40/48/56/64) |
| `config/user/single_user_*ses.yaml` | Data amount variants (2/4/8 sessions) |

### Training Commands

```bash
# BiLSTM baseline (150 epochs)
python -m emg2qwerty.train model=lstm_ctc

# BiLSTM with hop=48 (best config)
python -m emg2qwerty.train model=lstm_ctc transforms=log_spectrogram_hop48

# Screening run (40 epochs)
python -m emg2qwerty.train model=lstm_ctc trainer.max_epochs=40

# Channel ablation (e.g. 8ch = 8×33=264)
python -m emg2qwerty.train model=lstm_ctc module.in_features=264 trainer.max_epochs=40

# Data amount ablation (e.g. 8 sessions)
python -m emg2qwerty.train model=lstm_ctc user=single_user_8ses trainer.max_epochs=40

# Augmentation
python -m emg2qwerty.train model=lstm_ctc transforms=log_spectrogram_gaussian trainer.max_epochs=40
```

### Model Architecture

```
Input: (T, N, bands=2, channels=16, freq=33)
  → ChannelSlice(num_channels)
  → SpectrogramNorm(bands × channels)
  → MultiBandRotationInvariantMLP(in=528, out=[384]) per band
  → Flatten → (T, N, 768)
  → LSTMEncoder(num_features=768, hidden_size=384, num_layers=2)
  → Linear(768 → num_classes) → LogSoftmax → CTCLoss
```

---

## Environment Setup

```bash
source .venv/bin/activate
# If pkg_resources error: pip install setuptools==69.5.1
```

```bash
# VM setup
git clone -b han/LSTM --single-branch https://github.com/HOSH19/247A.git
gsutil -m cp -r gs://ec247a-emg2qwerty-data/data/ ~/247A/
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install -y cmake build-essential python3.10-dev
pip install -r requirements.txt && pip install -e .
```
