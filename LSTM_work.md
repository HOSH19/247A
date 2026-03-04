# LSTM Implementation Work Log

## Goal

Replace the TDSConvEncoder in the baseline pipeline with a BiLSTM encoder and systematically study the effect of architecture, preprocessing, and data characteristics on CER.

```
SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → [Encoder] → Linear → LogSoftmax → CTCLoss
```

MLP and CTC decoder kept identical to TDSConv baseline. Only the encoder and preprocessing are varied.

---

## Part 1: Architecture Experiments

Fixed: 16ch, 16 sessions, hop=16 (125Hz), standard augmentation (unless noted).

### Full Results Table

| Model | Total Params | Val CER | Test CER | Epochs | Notes |
|-------|-------------|---------|----------|--------|-------|
| TDSConv (baseline) | 5.3M | 22.55 | 24.18 | 40 | |
| TDSConv (baseline) | 5.3M | 18.94 | 22.17 | 150 | logs/2026-02-27/16-29-13 |
| BiLSTM (h=384, l=2) | 8.2M | 19.87 | 20.08 | 40 | |
| BiLSTM (h=384, l=3) | 11.7M | 20.87 | 24.08 | 40 | |
| BiLSTM (h=512, l=2) | 12.8M | 19.74 | 19.69 | 40 | |
| BiLSTM (h=512, l=3) | 19.1M | 17.88 | 21.55 | 40 | |
| BiLSTM + Transformer | 22.3M | 19.03 | 26.56 | 40 | screening |
| Conformer | 15.1M | 30.77 | 26.78 | 40 | screening |
| ConvLSTMConv (k=31) | 9.4M | 17.68 | 19.19 | 40 | hop=16 |
| TDS+BiLSTM hybrid | 13.0M | — | — | 40 | (no 40ep run) |
| BiLSTM (h=384, l=2) | 8.2M | **14.55** | 15.76 | 150 | logs/2026-02-28/00-52-40 |
| BiLSTM (h=512, l=3) | 19.1M | 15.91 | 22.80 | 150 | best ep135 |
| TDS+BiLSTM hybrid | 13.0M | 14.58 | 15.47 | 150 | warm init from TDSConv ep38 |
| BiLSTM + Transformer | 22.3M | 14.67 | 17.25 | 150 | best ep129 |
| BiLSTM hop=48 | 8.2M | 13.98 | **14.52** | 150 | best preprocessing |
| ConvLSTMConv hop=48 | 9.4M | 14.49 | 17.53 | 150 | best ep110 |
| BiLSTM win=4000 hop=48 | 8.2M | 17.61 | 16.08 | 40 | window ablation |
| BiLSTM win=16000 hop=48 | 8.2M | 15.86 | 16.19 | 40 | window ablation |
| BiLSTM pad=[900,100] hop=48 | 8.2M | 15.77 | 16.06 | 40 | padding ablation |
| BiLSTM pad=[3600,400] hop=48 | 8.2M | 17.39 | 17.22 | 40 | padding ablation |
| BiLSTM win=16000 pad=[900,100] hop=48 | 8.2M | **15.13** | 16.27 | 40 | best combo |
| BiLSTM win=16000 pad=[900,100] hop=48 | 8.2M | **13.65** | **14.93** | 150 | best config full run |

---

### Experiment 1: BiLSTM vs TDSConv

**Motivation:** TDSConv uses fixed-width convolutional kernels — each output frame can only see a local 62ms window (kernel=32, hop=2). EMG keystroke patterns involve muscle activations that span variable timescales and benefit from knowing both past and future context. A bidirectional recurrent model sees the entire sequence and can model these long-range dependencies directly.

**Result:** BiLSTM (val 14.55) outperforms TDSConv (18.94) by **−4.39 val CER** at 150 epochs.

**Insight:** The improvement is large and consistent. Full-sequence bidirectional context is clearly beneficial for EMG keystroke decoding. TDS's fixed local receptive field is a genuine architectural bottleneck for this task.

---

### Experiment 2: TDS+BiLSTM Hybrid

**Motivation:** TDS and BiLSTM should be complementary — TDS extracts local frequency-temporal features efficiently, while BiLSTM models global sequence context. Stacking them might give the best of both.

**Architecture:**
```
Flatten → TDSConvEncoder(kernel=32) → LSTMEncoder(h=384, l=2) → Linear
```
Warm-initialized from a pre-trained TDSConv checkpoint (ep38) to preserve local feature representations.

**Result:** TDS+BiLSTM (14.58) ≈ pure BiLSTM (14.55). **No meaningful improvement.**

**Insight:** Two interpretations:
1. BiLSTM already learns local temporal patterns through its recurrent connections — TDS preprocessing is redundant
2. TDS's temporal reduction (T → T−124) discards a small amount of edge context that BiLSTM would otherwise use

Either way, simply prepending TDS to BiLSTM is not a useful direction. The hybrid model adds 4.8M params for zero gain.

---

### Experiment 3: Scale up BiLSTM

**Motivation:** The baseline h=384, l=2 config may simply be underpowered. Before exploring architectural variants, check whether raw capacity is the bottleneck by running a 2×2 factorial over hidden size and depth.

**2×2 Factorial @ 40 epochs:**

| | l=2 | l=3 |
|---|---|---|
| **h=384** | 19.87 / 20.08 | 20.87 / 24.08 |
| **h=512** | 19.74 / 19.69 | **17.88 / 21.55** |

*(val CER / test CER)*

Positive interaction effect observed — scaling both h and l together yields the best 40-epoch result (17.88). Ran full 150-epoch run with h=512, l=3.

**Result: val CER 15.91, test CER 22.80 — worse than h=384, l=2 (14.55/15.76) at 150 epochs.**

**Insight:** The larger model overfits on the single-user dataset. Despite a promising 40-epoch screening result, the capacity gains don't hold at convergence. h=384, l=2 is already near the sweet spot for this data regime. **The bottleneck is data size, not model capacity.**

---

### Experiment 4: BiLSTM + Self-Attention (Transformer)

**Motivation:** BiLSTM processes sequences step-by-step — information about distant timesteps must propagate through intermediate hidden states. Transformer self-attention allows direct pairwise comparison of any two frames, which could help the model recognize keystroke patterns that share long-range context (e.g., co-articulation between adjacent keystrokes).

**Architecture:**
```
Flatten → LSTMEncoder(h=384, l=2) → TransformerEncoder(layers=2, nhead=8, ffn=3072) → Linear
```
No positional encoding — LSTM output already encodes position implicitly via recurrent state.

**Screening (40 epochs):** val CER 19.03 — slower to converge than pure BiLSTM (19.87 at ep40).

**Full run (150 epochs):** val CER **14.67**, test CER **17.25** — essentially identical to pure BiLSTM (14.55/15.76).

**Insight:** Adding self-attention on top of BiLSTM converges to the same solution but slower, and with worse test CER (17.25 vs 15.76). Two interpretations:
1. BiLSTM's bidirectional hidden states already capture sufficient cross-timestep context — attention has nothing new to add
2. The extra 14M parameters (22.3M vs 8.2M) overfit on the small single-user dataset

Consistent with Exp 3: the data regime limits what any architectural addition can achieve.

---

### Experiment 5: Conformer

**Motivation:** The Conformer (Gulati et al., 2020) combines self-attention and depthwise convolution in a sandwich structure (FFN → MHSA → DepthwiseConv → FFN), achieving state-of-the-art on speech recognition tasks. The combination of local (conv) and global (attention) context modeling seems well-suited for EMG sequences.

**Architecture:**
```
Flatten → torchaudio.Conformer(input_dim=768, num_heads=8, ffn_dim=1024, num_layers=2, kernel=31) → Linear
```
Config tuned to 15.1M params (num_layers=2, ffn_dim=1024) to avoid extreme overfitting.

**Screening (40 epochs):** val CER **30.77**, test CER **26.78** — dramatically worse than BiLSTM (19.87/20.08).

**Insight:** Conformer fails badly on this task despite being a strong architecture for speech. The self-attention mechanism requires large amounts of data to learn meaningful cross-timestep relationships — on a single-user dataset with 16 sessions, it simply overfits. The 15.1M parameter count is already limiting, and even at 40 epochs, performance is far below the 8.2M BiLSTM. **Self-attention is the wrong inductive bias for this data regime.**

---

### Experiment 6: Conv → BiLSTM → Conv (ConvLSTMConv)

**Motivation:** Inspired by the Conformer's sandwich structure (Conv-Attention-Conv), replace self-attention with BiLSTM. Depthwise conv blocks on both sides of the BiLSTM could capture local temporal patterns (~248ms window) that complement the BiLSTM's global context modeling — without the data-hungry attention mechanism.

**Architecture:**
```
Flatten → ConvBlock(k=31) → LSTMEncoder(h=384, l=2) → ConvBlock(k=31) → Linear
```
ConvBlock: LayerNorm → DepthwiseConv1d(k=31, groups=C) → GELU → PointwiseConv1d(1×1) → residual

**Parameter breakdown (9.4M total):**
| Component | Params |
|---|---|
| Pre-ConvBlock | 0.617M |
| LSTMEncoder (h=384, l=2) | 7.681M |
| Post-ConvBlock | 0.617M |
| Classifier | 0.045M |

**Screening @ hop=16 (40 epochs):** val CER **17.68**, test CER **19.19** — noticeably better than pure BiLSTM at 40ep (19.87/20.08) with only 1.2M extra params.

**Full run @ hop=48 (150 epochs):** val CER **14.49**, test CER **17.53** — val matches BiLSTM hop=16 (14.55) but test CER is significantly worse (17.53 vs 14.52 for BiLSTM hop=48).

**Insight:** ConvLSTMConv converges faster than BiLSTM (promising 40ep result) but overfits at convergence — the additional 1.2M conv parameters are enough to tip the balance on this small dataset. The conv blocks appear redundant: BiLSTM already learns local temporal structure through its recurrent connections (same conclusion as Exp 2). Each architectural addition that increases parameter count hurts generalization in this data-limited regime.

---

### Experiment 7: Window Size and Padding

**Motivation:** The baseline uses `window_length=8000` (4s) with `padding=[1800,200]` (900ms past / 100ms future context), inherited without ablation. Larger windows give BiLSTM more sequence context per forward pass; padding provides extra context beyond the prediction window at boundaries. Both could affect CER.

**Setup:** BiLSTM h=384, l=2, hop=48, 40 epochs, gradient_clip_val=1.0. Baseline window=8000, pad=[1800,200] → val 17.01, test 17.59.

**Window size ablation (pad=[1800,200] fixed):**

| Window | Duration | Frames (hop=48) | Val CER | Test CER |
|--------|----------|-----------------|---------|----------|
| 4000 | 2s | ~104 | 17.61 | 16.08 |
| **8000** | **4s (baseline)** | **~208** | **17.01** | **17.59** |
| **16000** | **8s** | **~417** | **15.86** | **16.19** |

**Padding ablation (window=8000 fixed):**

| Padding | Past context | Future context | Val CER | Test CER |
|---------|-------------|----------------|---------|----------|
| [900, 100] | 450ms | 50ms | **15.77** | **16.06** |
| **[1800, 200]** | **900ms (baseline)** | **100ms** | **17.01** | **17.59** |
| [3600, 400] | 1800ms | 200ms | 17.39 | 17.22 |

**Combo experiments:**

| Window | Padding | Val CER | Test CER |
|--------|---------|---------|----------|
| 4000 | [3600, 400] | 17.10 | 23.17 |
| **16000** | **[900, 100]** | **15.13** | **16.27** |
| 16000 | [3600, 400] | 16.08 | 17.25 |

**Best: win=16000, pad=[900,100] → val 15.13 @ 40ep → val 13.65, test 14.93 @ 150ep (best ep120).**

**Insight:**
- **Larger window consistently helps**: 8s window (val 15.86) > 4s (17.01) > 2s (17.61). BiLSTM benefits from longer sequences because it can model more keystroke context per forward pass.
- **Smaller padding is better**: pad=[900,100] (val 15.77) beats baseline pad=[1800,200] (17.01). Counterintuitive — extra padding adds noisy boundary context that the model must learn to ignore.
- **pad=[3600,400] hurts**: Too much padding degrades performance. The model is distracted by context far from the prediction window.
- **Best combo**: win=16000 + pad=[900,100] → val 15.13, nearly −2 over baseline at 40ep.
- win=4000 + pad=[3600,400] shows severe test overfitting (23.17) — short window with large padding is the worst combination.

---

## Part 2: Preprocessing Ablation

Fixed: BiLSTM h=384, l=2 (8.2M), 16ch, 16 sessions.

### Experiment 8: Sampling Rate (hop_length)

**Motivation:** The baseline uses `hop_length=16`, downsampling the 2kHz EMG signal to 125 spectrogram frames/sec. This choice was inherited from the original codebase without ablation. Higher temporal resolution preserves more detail but creates longer sequences; lower resolution may discard useful information but could be easier for BiLSTM to model. Worth exploring whether 125Hz is actually optimal.

**Implementation:** Created `config/transforms/log_spectrogram_hop{N}.yaml` for N ∈ {8, 16, 24, 32, 40, 48, 56, 64}. `n_fft=64` and `in_features=528` unchanged — only temporal resolution varies.

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

**Full run at hop=48 (150 epochs):**

| Model | Total Params | Val CER | Test CER | Epochs |
|-------|-------------|---------|----------|--------|
| BiLSTM hop=16 (baseline) | 8.2M | 14.55 | 15.76 | 150 |
| **BiLSTM hop=48** | **8.2M** | **13.98** | **14.52** | **150** |

**Insight:** Counterintuitively, lower temporal resolution (hop=32~64) consistently outperforms the baseline (hop=16). Two reasons:
1. **Shorter sequences**: hop=48 produces 3× fewer frames per window → better gradient flow through BiLSTM, less vanishing gradient
2. **Noise reduction**: EMG keystroke patterns operate on ~50–200ms timescales — 41.7Hz is more than sufficient resolution, while 125Hz likely captures inter-frame noise

hop=8 (250Hz) is clearly worst: longest sequences + most noise. The sweet spot is **hop=48 (41.7Hz)**, confirmed at both 40ep and 150ep.

---

### Experiment 9: Data Augmentation

**Motivation:** With only 16 training sessions for one user, overfitting is a real concern. Standard augmentation techniques may improve generalization by diversifying training examples. The baseline already uses RandomBandRotation, TemporalAlignmentJitter, and SpecAugment — but these are EMG-specific. Testing whether general signal augmentations (noise, amplitude variation) provide additional benefit.

**Implementation:** Added two new classes to `transforms.py`:
- `GaussianNoise(std=0.1)` — adds Gaussian noise to log-spectrogram values (post-spectrogram)
- `AmplitudeScale(min=0.7, max=1.3)` — randomly scales raw EMG amplitude before spectrogram (simulates inter-session muscle activation variability)

| Augmentation | Total Params | Val CER | Test CER |
|---|---|---|---|
| Baseline (RandBandRot + TempJitter + SpecAugment) | 8.2M | 19.87 | 20.08 |
| + GaussianNoise (std=0.1) | 8.2M | 20.16 | 20.34 |
| + AmplitudeScale (×0.7~1.3) | 8.2M | 21.02 | 21.98 |

**Insight:** Both augmentations slightly hurt at 40 epochs. The baseline pipeline already contains three augmentation stages — adding more regularization slows convergence on this small dataset without improving generalization. The existing SpecAugment (time + freq masking) combined with band rotation appears to be sufficient for this data regime.

---

## Part 3: Data Ablation

Fixed: BiLSTM h=384, l=2 (8.2M), hop=16, 40 epochs.

### Experiment 10: Electrode Channel Ablation

**Motivation:** The device uses 16 electrode channels per band (32 total). Fewer channels = simpler, cheaper hardware. Understanding how many channels are actually necessary could inform future hardware design. Hypothesis: nearby electrodes may capture redundant signals, so some reduction should be tolerable.

**Implementation:** Added `ChannelSlice` module to `modules.py` — selects first N channels per band as the first layer. Controlled via `module.in_features` override (`in_features = num_channels × 33`).

| Channels per band | in_features | Total Params | Val CER | Test CER |
|---|---|---|---|---|
| 16 (full) | 528 | 8.2M | 19.87 | 20.08 |
| 8 | 264 | 8.2M | 25.88 | 26.35 |
| 4 | 132 | 8.2M | 36.53 | 37.97 |
| 2 | 66 | 8.2M | 66.59 | 67.80 |
| 1 | 33 | 8.2M | 88.04 | 86.90 |

**Insight:** Performance degrades monotonically and steeply — CER roughly doubles with every halving of channels. The hypothesis that nearby electrodes are redundant is clearly wrong: each electrode captures spatially distinct muscle activation patterns that the model relies on non-redundantly. At 2ch/1ch the model barely learns at all, suggesting that even a small number of channels is insufficient to discriminate between finger movements. All 16 channels are needed for competitive performance.

---

### Experiment 11: Training Data Amount

**Motivation:** The single-user dataset has 16 training sessions (collected across multiple days). Collecting many sessions is expensive and time-consuming — if the model works well with fewer sessions, data collection burden is reduced. Also directly tests the data-bottleneck hypothesis: if less data dramatically hurts performance, then architecture improvements are fundamentally limited by data size.

**Implementation:** Created `config/user/single_user_{2,4,8}ses.yaml` configs with subsets of the 16 training sessions (first N sessions in chronological order). Val/test sessions kept identical across all runs.

| Train sessions | Fraction | Total Params | Val CER | Test CER |
|---|---|---|---|---|
| 2 | 12.5% | 8.2M | ~100 (fails) | ~100 |
| 4 | 25% | 8.2M | ~100 (fails) | ~100 |
| 8 | 50% | 8.2M | 36.97 | 33.46 |
| 16 (full) | 100% | 8.2M | **19.87** | **20.08** |

**Insight:** There is a sharp threshold between 4 and 8 sessions. Below 8 sessions, the model immediately overfits — training loss decreases while val CER stays at ~100, meaning it memorizes training data without any generalization. With 8 sessions the model learns something useful (36.97) but is far from the full-data result (19.87). All 16 sessions are needed for competitive performance. This **strongly confirms the data-bottleneck hypothesis** from the architecture experiments: the fundamental limit is not model architecture but available training data.

---

## Implementation Details

### Key Files

| File | Description |
|------|-------------|
| `emg2qwerty/modules.py` | `LSTMEncoder`, `ChannelSlice`, `ConvBlock` classes |
| `emg2qwerty/lightning.py` | `LSTMCTCModule`, `LSTMTransformerCTCModule`, `ConformerCTCModule`, `ConvLSTMConvCTCModule` |
| `emg2qwerty/transforms.py` | `GaussianNoise`, `AmplitudeScale` classes added |
| `config/model/lstm_ctc.yaml` | BiLSTM config (h=384, l=2) |
| `config/model/lstm_transformer_ctc.yaml` | BiLSTM + Transformer config |
| `config/model/conformer_ctc.yaml` | Conformer config (15.1M) |
| `config/model/conv_lstm_conv_ctc.yaml` | ConvLSTMConv config (9.4M, k=31) |
| `config/transforms/log_spectrogram_hop*.yaml` | Sampling rate variants (hop 8/16/24/32/40/48/56/64) |
| `config/user/single_user_*ses.yaml` | Data amount variants (2/4/8 sessions) |

### Training Commands

```bash
# BiLSTM baseline (150 epochs)
python -m emg2qwerty.train model=lstm_ctc

# BiLSTM with hop=48 (best config)
python -m emg2qwerty.train model=lstm_ctc transforms=log_spectrogram_hop48

# ConvLSTMConv with hop=48
python -m emg2qwerty.train model=conv_lstm_conv_ctc transforms=log_spectrogram_hop48

# Conformer (40ep screening)
python -m emg2qwerty.train model=conformer_ctc trainer.max_epochs=40

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
