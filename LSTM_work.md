# LSTM Implementation Work Log

## Goal

Replace the TDSConvEncoder in the baseline pipeline with a BiLSTM encoder and systematically study the effect of architecture, preprocessing, and data characteristics on CER.

```
SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → [Encoder] → Linear → LogSoftmax → CTCLoss
```

MLP and CTC decoder kept identical to TDSConv baseline. Only the encoder and preprocessing are varied.

---

## Overall Results Summary

### Architecture Experiments (150 epochs, hop=16 baseline preprocessing)

| Model | Val CER | Test CER | Epochs | Notes |
|-------|---------|----------|--------|-------|
| TDSConv (baseline) | 18.94 | 22.17 | 150 | logs/2026-02-27/16-29-13 |
| BiLSTM (h=384, l=2) | **14.55** | **15.76** | 150 | logs/2026-02-28/00-52-40 |
| TDS+BiLSTM hybrid | 14.58 | 15.47 | 150 | warm init ep38 |
| BiLSTM (h=512, l=3) | 15.91 | 22.80 | 150 | best ep135 |
| BiLSTM + Transformer | 14.67 | 17.25 | 150 | best ep129 |

### Preprocessing: Sampling Rate (40 epochs, BiLSTM h=384 l=2)

| hop_length | Effective rate | Val CER | Test CER |
|-----------|----------------|---------|----------|
| 8 | 250 Hz | 26.47 | 25.14 |
| 16 | 125 Hz (baseline) | 19.87 | 20.08 |
| 24 | 83 Hz | 18.76 | 18.50 |
| 32 | 62.5 Hz | 17.50 | 16.99 |
| 40 | 50 Hz | 18.17 | 18.31 |
| **48** | **41.7 Hz** | **17.01** | **17.59** |
| 56 | 35.7 Hz | 17.92 | 18.22 |
| 64 | 31.25 Hz | 17.68 | 17.53 |

### Best Config Full Run (150 epochs, BiLSTM h=384 l=2, hop=48)

| Model | Val CER | Test CER | Epochs |
|-------|---------|----------|--------|
| BiLSTM hop=16 (baseline) | 14.55 | 15.76 | 150 |
| **BiLSTM hop=48** | **13.98** | **14.52** | **150** |

### Preprocessing: Data Augmentation (40 epochs, BiLSTM h=384 l=2, hop=16)

| Augmentation | Val CER | Test CER |
|---|---|---|
| Baseline (none extra) | 19.87 | 20.08 |
| + GaussianNoise (std=0.1) | 20.16 | 20.34 |
| + AmplitudeScale (×0.7~1.3) | 21.02 | 21.98 |

### Data Ablation: Electrode Channels (40 epochs)

| Channels per band | Val CER |
|---|---|
| 16 (full) | 19.87 |
| 8 | 25.88 |
| 4 | 36.53 |
| 2 | 66.59 |
| 1 | 88.04 |

### Data Ablation: Training Sessions (40 epochs)

| Train sessions | Fraction | Val CER |
|---|---|---|
| 2 | 12.5% | ~100 (fails) |
| 4 | 25% | ~100 (fails) |
| 8 | 50% | 36.97 |
| 16 (full) | 100% | **19.87** |

---

## Part 1: Architecture Experiments

### Experiment 1: BiLSTM vs TDSConv

**Motivation:** TDSConv only sees a fixed 62ms local window per step. EMG keystroke patterns require longer temporal context — a full-sequence model should do better.

**Architecture:**
```
Input: (T, N, 2, 16, 33)
  → SpectrogramNorm(32)
  → MultiBandRotationInvariantMLP(in=528, out=384) per band
  → Flatten → (T, N, 768)
  → LSTMEncoder(num_features=768, hidden_size=384, num_layers=2, dropout=0.1)
  → Linear(768 → num_classes) → LogSoftmax
```

**Result:** BiLSTM (14.55) significantly outperforms TDSConv (18.94), **−4.39 val CER**.

**Insight:** Full-sequence bidirectional context is clearly beneficial for this task. The fixed receptive field of TDS is a real bottleneck.

---

### Experiment 2: TDS+BiLSTM Hybrid

**Motivation:** Stack TDS (local feature extraction) before BiLSTM (global context), expecting each to handle what the other cannot.

**Architecture:**
```
Flatten → TDSConvEncoder(kernel=32) → LSTMEncoder(hidden=384, layers=2) → Linear
```

**Result:** TDS+BiLSTM (14.58) ≈ BiLSTM (14.55), **no meaningful improvement**.

**Insight:** BiLSTM already captures local patterns through its recurrent connections — TDS preprocessing is redundant. TDS's temporal reduction (T → T−124) may also slightly hurt by discarding edge context.

---

### Experiment 3: Scale up BiLSTM

**Motivation:** Check if raw model capacity is the bottleneck.

**2×2 Factorial @ 40 epochs:**

| | layers=2 | layers=3 |
|---|---|---|
| **hidden=384** | 22.53 / — | 20.87 / 24.08 |
| **hidden=512** | 19.74 / 19.69 | **17.88 / 21.55** |

*(val CER / test CER)*

Factorial showed positive interaction — scaling both yields best 40-epoch result. Full 150-epoch run with h=512, l=3:

**Result: val CER 15.91, test CER 22.80 — worse than h=384, l=2 (14.55) at 150ep.**

**Insight:** Larger model overfits on single-user data. h=384, l=2 is near the capacity sweet spot. The bottleneck is data, not model capacity.

---

### Experiment 4: BiLSTM + Self-Attention (Transformer)

**Motivation:** BiLSTM may struggle with direct long-range dependencies. Stacking Transformer layers on top lets the model directly attend to any pair of timesteps.

**Architecture:**
```
Flatten → LSTMEncoder(h=384, l=2) → TransformerEncoder(layers=2, nhead=8, ffn=3072) → Linear
```
No positional encoding — LSTM output already encodes position implicitly.

**Screening (40 epochs):** val CER 19.03 — slower to converge than pure BiLSTM.

**Full run (150 epochs):** val CER **14.67**, test CER **17.25** — essentially identical to pure BiLSTM (14.55).

**Insight:** BiLSTM's bidirectional hidden states already capture sufficient context. Extra attention parameters don't help on small single-user data. Consistent with data-bottleneck finding.

---

## Part 2: Preprocessing Ablation

### Experiment 5: Sampling Rate (hop_length)

**Motivation:** The baseline uses `hop_length=16` (2kHz → 125 frames/sec). Is this the right temporal resolution?

**Implementation:** Created transform configs `log_spectrogram_hop{8,16,24,32,40,48,56,64}.yaml` varying only `hop_length`. `n_fft=64` and `in_features=528` unchanged.

**Results:** See summary table above. Sweet spot is **hop=48 (41.7 Hz)**.

**Insight:** Lower temporal resolution (hop=32~64) outperforms baseline (hop=16). Two reasons:
1. **Shorter sequences**: fewer frames → better BiLSTM gradient flow
2. **Noise reduction**: EMG keystroke patterns operate on ~50–200ms timescales — 125Hz captures temporal detail that is mostly noise

hop=8 (250Hz) is worst — longer sequences hurt, extra detail adds noise.

**Full run at hop=48 (150 epochs): val CER 13.98, test CER 14.52 — new best overall.**

---

### Experiment 6: Data Augmentation

**Motivation:** The single-user dataset is small and the model overfits. Test whether additional augmentation improves generalization.

**Implementation:** Added two new transform classes to `transforms.py`:
- `GaussianNoise(std=0.1)` — applied after LogSpectrogram
- `AmplitudeScale(min=0.7, max=1.3)` — applied before LogSpectrogram (raw EMG)

**Results:** See summary table above. Both slightly hurt performance.

**Insight:** The baseline already has substantial augmentation (RandomBandRotation + TemporalAlignmentJitter + SpecAugment). Adding more regularization slows convergence without improving generalization on this small dataset.

---

## Part 3: Data Ablation

### Experiment 7: Electrode Channel Ablation

**Motivation:** The model uses 16 electrode channels per band. How many are actually needed?

**Implementation:** Added `ChannelSlice` module to `modules.py` — selects first N channels per band. Controlled via `module.in_features` override (`in_features = num_channels × 33`).

**Results:** See summary table above.

**Insight:** Performance degrades monotonically and steeply as channels are reduced. CER roughly doubles each time channels are halved (16→8→4). At 2ch/1ch the model barely learns. All 16 channels carry spatially distinct, non-redundant EMG information.

---

### Experiment 8: Training Data Amount

**Motivation:** How many training sessions are actually needed?

**Implementation:** Created `config/user/single_user_{2,4,8}ses.yaml` configs with session subsets. Val/test sessions identical across all runs.

**Results:** See summary table above.

**Insight:** Sharp threshold between 4 and 8 sessions — below 8, the model completely fails to generalize (immediate overfitting, val CER ~100). All 16 sessions are needed for competitive performance. Strongly confirms the data-bottleneck hypothesis.

---

## Implementation Details

### Key Files

| File | Description |
|------|-------------|
| `emg2qwerty/modules.py` | `LSTMEncoder`, `ChannelSlice` classes |
| `emg2qwerty/lightning.py` | `LSTMCTCModule`, `LSTMTransformerCTCModule` |
| `config/model/lstm_ctc.yaml` | BiLSTM config (h=384, l=2) |
| `config/model/lstm_transformer_ctc.yaml` | BiLSTM + Transformer config |
| `config/transforms/log_spectrogram_hop*.yaml` | Sampling rate variants |
| `config/user/single_user_*ses.yaml` | Data amount variants |

### Training Commands

```bash
# BiLSTM baseline
python -m emg2qwerty.train model=lstm_ctc

# BiLSTM with hop=48 (best config)
python -m emg2qwerty.train model=lstm_ctc transforms=log_spectrogram_hop48

# Channel ablation (e.g. 8ch)
python -m emg2qwerty.train model=lstm_ctc module.in_features=264

# Data amount ablation (e.g. 8 sessions)
python -m emg2qwerty.train model=lstm_ctc user=single_user_8ses

# Screening run (40 epochs)
python -m emg2qwerty.train model=lstm_ctc trainer.max_epochs=40
```

### Model Architecture

```
Input: (T, N, bands=2, channels=16, freq=33)
  → ChannelSlice(num_channels)          # channel ablation layer
  → SpectrogramNorm(bands × channels)
  → MultiBandRotationInvariantMLP(in=528, out=[384]) per band
  → Flatten → (T, N, 768)
  → LSTMEncoder(num_features=768, hidden_size=384, num_layers=2)
      BiLSTM: (T, N, 768) → (T, N, 768)
  → Linear(768 → num_classes)
  → LogSoftmax → CTCLoss
```

---

## Environment Setup

```bash
source .venv/bin/activate
# If pkg_resources error: pip install setuptools==69.5.1
```

### VM Setup Steps

```bash
git clone -b han/LSTM --single-branch https://github.com/HOSH19/247A.git
gsutil -m cp -r gs://ec247a-emg2qwerty-data/data/ ~/247A/
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install -y cmake build-essential python3.10-dev
pip install -r requirements.txt && pip install -e .
```
