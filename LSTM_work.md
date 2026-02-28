# LSTM Implementation Work Log

## Goal
Replace the TDSConvEncoder in the baseline pipeline with a BiLSTM encoder
to compare performance (CER) against the TDS CNN baseline (~30 val CER).

---

## Files Modified / Created

### 1. `emg2qwerty/modules.py` — `LSTMEncoder` class added

```python
class LSTMEncoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=False,  # TNC format
        )
        self.fc = nn.Linear(hidden_size * 2, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(inputs)  # (T, N, hidden_size * 2)
        return self.fc(x)         # (T, N, num_features)
```

- Bidirectional LSTM: captures both past and future context
- `fc` layer projects back to `num_features` (768) to keep output shape consistent with TDS baseline
- `batch_first=False`: preserves TNC format used throughout the codebase

### 2. `emg2qwerty/lightning.py` — `LSTMCTCModule` class added

- Added `LSTMEncoder` to imports
- Added `LSTMCTCModule(pl.LightningModule)` — mirrors `TDSConvCTCModule` exactly,
  with `TDSConvEncoder` replaced by `LSTMEncoder`
- New `__init__` params: `hidden_size`, `num_layers`, `dropout`
- `_step` logic is identical to `TDSConvCTCModule`
  (T_diff handles temporal length difference, though LSTM preserves T so T_diff=0)

### 3. `config/model/lstm_ctc.yaml` — new config file created

```yaml
# @package _global_
module:
  _target_: emg2qwerty.lightning.LSTMCTCModule
  in_features: 528
  mlp_features: [384]
  hidden_size: 384
  num_layers: 2
  dropout: 0.1

datamodule:
  _target_: emg2qwerty.lightning.WindowedEMGDataModule
  window_length: 8000
  padding: [1800, 200]
```

### 4. `.github/workflows/testing.yml` — CI trigger changed

```yaml
# Before
on: [push]

# After
on:
  push:
    branches: [main]
```

CI only runs on `main` branch pushes — no more failure emails on experiment branches.

---

## Model Architecture (LSTM)

```
Input: (T, N, 2, 16, 33)
  → SpectrogramNorm(32)
  → MultiBandRotationInvariantMLP(in=528, out=384) per band
  → Flatten  →  (T, N, 768)
  → LSTMEncoder(num_features=768, hidden_size=384, num_layers=2, dropout=0.1)
      BiLSTM: (T, N, 768) → (T, N, 768)   [384*2=768, projected back via fc]
  → Linear(768 → num_classes)
  → LogSoftmax
```

---

## Training Command

```bash
python -m emg2qwerty.train model=lstm_ctc
```

Overriding hyperparameters example:
```bash
python -m emg2qwerty.train model=lstm_ctc module.hidden_size=512 module.num_layers=3
```

---

## Environment Setup

### VM Setup Steps

```bash
# 1. Clone repo & download data
git clone -b han/LSTM --single-branch https://github.com/HOSH19/247A.git
gsutil -m cp -r gs://ec247a-emg2qwerty-data/data/ ~/247A/

# 2. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install system dependencies required to build kenlm
sudo apt-get install -y cmake build-essential python3.10-dev

# 4. Install Python packages
pip install -r requirements.txt
pip install -e .
```

### Known Issues

- Running `pip install --upgrade pip wheel setuptools` upgrades setuptools to 82.x,
  which drops `pkg_resources`, causing `pytorch-lightning` import to fail with
  `ModuleNotFoundError: No module named 'pkg_resources'`
- Fix: `pip install setuptools==69.5.1` (pin to version specified in requirements.txt)

---

## Baseline Comparison

| Model | Encoder | Temporal context | Val CER | Epochs | Log |
|-------|---------|-----------------|---------|--------|-----|
| TDSConv (baseline) | TDS CNN (4 blocks, kernel=32) | Fixed 124 samples (~62ms) | **18.94** | 150 (full) | logs/2026-02-27/16-29-13 |
| BiLSTM (ours) | Bidirectional LSTM | Full sequence | **14.55** | 150 (full) | logs/2026-02-28/00-52-40 |

---

## Planned Experiments

1. BiLSTM (hidden=384, layers=2) — current config
2. Vary `num_layers`: 1, 2, 3, 4
3. Vary `hidden_size`: 256, 384, 512
4. Unidirectional vs Bidirectional
5. CNN+LSTM hybrid (TDSConvEncoder → LSTMEncoder stacked)
