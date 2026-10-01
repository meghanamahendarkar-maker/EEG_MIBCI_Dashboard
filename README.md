# Motor Imagery EEG Classification: MiniRocket vs. Hybrid CNN-LSTM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/NeuroImage-2026-1e3a8a.svg)](https://doi.org/10.1016/j.neuroimage.2026.121816)

This repository provides an open-source, fully reproducible implementation of the research paper:
> **"Motor imagery EEG signal classification using minimally random convolutional kernel transform and hybrid deep learning"**  
> *Jamal Hwaidi & Mohamed Chahine Ghanem*  
> **NeuroImage 328 (2026) 121816**. [DOI: 10.1016/j.neuroimage.2026.121816](https://doi.org/10.1016/j.neuroimage.2026.121816)

---

## 🔬 System Overview

The framework investigates computationally efficient, clinical-grade decoding of 4-class Motor Imagery (MI) electroencephalography (EEG):
- **T1**: Left Fist ($L$)
- **T2**: Right Fist ($R$)
- **T3**: Both Fists ($BLR$)
- **T4**: Both Feet ($BF$)

It rigorously compares two competing classification paradigms:
1. **Pipeline A (Deterministic Transform + Linear Classification):**
   - **MiniRocket** feature extractor: 10,000 dilated 1D kernels ($L=9$, dilations $\le 28$).
   - Proportion of Positive Values (**PPV**) pooling operator.
   - Closed-form **Ridge Regression** classifier (no backpropagation, no gradient vanishing).
2. **Pipeline B (Deep Learning End-to-End Baseline):**
   - **13-Layer Hybrid CNN-LSTM**: 2 Conv1D layers (16 & 32 filters) $\to$ Max Pooling $\to$ 100-unit LSTM $\to$ Dense layers $\to$ Sigmoid/Softmax.
   - Trained with Adam optimizer ($\eta = 10^{-5}$) and L2 weight decay ($0.01$).

```
Raw EEG (160 Hz) ──► Anti-Aliased Resampling (128 Hz) ──► CAR ──► Bandpass (8-30 Hz) ──► ICA
                                                                                            │
                              ┌─────────────────────────────────────────────────────────────┘
                              ▼
            5 Symmetric Motor Cortex Electrode Pairs
                 (Serially Connected Sample: 1280 pts)
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   [MiniRocket + Ridge]              [13-Layer CNN-LSTM]
   • 10,000 kernels (dilations <= 28) • Conv1D (16, 32)
   • PPV-only pooling                • LSTM (100 units)
   • Closed-form solve               • Adam + L2 decay
             │                                 │
             ▼                                 ▼
   Predictions (T1-T4)               Predictions (T1-T4)
```

---

## 📊 Summary of Published Results

### Classification Accuracy
| Benchmark Dataset | Classes | MiniRocket + Ridge | Hybrid CNN-LSTM | Prior SOTA |
| :--- | :--- | :--- | :--- | :--- |
| **PhysioNet EEGMMIDB** (S1–S10) | 4 tasks | **98.63%** | **98.06%** | 97.88% (*Li et al., 2023*) |
| **BCI Competition IV-2a** | 4 tasks | **92.57%** | **92.32%** | 88.03% (*Salami et al., 2022*) |

### Computational Complexity & CPU Latency
| Metric | MiniRocket + Ridge | Hybrid CNN-LSTM | Efficiency Advantage |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | ~40,000 | ~250,000 | **6.25× fewer parameters** |
| **Total CPU Train Time** | **6.0 minutes** | **150.0 minutes** | **25× faster training** |
| **Inference Latency / Sample** | **0.6 ms** | **8.0 ms** | **13.3× lower latency** |
| **Trial Latency (9 samples)** | **5.4 ms** | **72.3 ms** | Sub-10ms embedded response |
| **Peak RAM Consumption** | **1.2 GB** | **2.8 GB** | **2.3× lower memory footprint** |

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/eeg-mibci-minirocket.git
cd eeg-mibci-minirocket
pip install -e .
```

### 2. Fast 10-Second Demo
Verify data generation, MiniRocket transform, and CNN-LSTM forward pass:
```bash
python scripts/run_demo.py
```

### 3. Run the End-to-End Pipeline & Generate Figures
Run the automated benchmarking pipeline across subjects and generate Figures 6, 7, 8, 9:
```bash
# Fast demonstration mode
python scripts/run_pipeline.py --mode fast_demo --subjects 3 --epochs 10

# Ablation studies (CNN-GRU, SVM, Logistic Regression)
python scripts/run_pipeline.py --mode ablations --subjects 2

# Full PhysioNet download and benchmark (requires internet connection)
python scripts/run_pipeline.py --mode physionet --subjects 10 --epochs 100
```

All generated publication plots are stored in `results/`:
- `results/figure_6_subject_accuracies.png`: Per-subject accuracy distributions.
- `results/figure_7_confusion_matrices.png`: Mean 4-class confusion matrices.
- `results/figure_8_roc_curves.png`: Multi-class ROC curves & AUC.
- `results/figure_9_training_curves.png`: Epoch vs. Loss and Accuracy curves.
- `results/benchmark_comparison.png`: Parameter and latency bar charts.
- `results/computational_benchmark.json`: Raw CPU timing and memory benchmarks.

---

## 🖥️ Interactive Web Dashboard

Launch the Streamlit neuroengineering dashboard:
```bash
streamlit run src/app/app.py
```

The interactive application includes:
- **Signal & Spectrum Explorer**: Interactive inspection of $\mu$ (8–14 Hz) and $\beta$ (14–30 Hz) sensorimotor rhythm desynchronization (ERD/ERS).
- **Live Classifier Arena**: Compare MiniRocket vs. CNN-LSTM on individual trials with real-time probability distributions and latency timers.
- **Benchmark & Figures**: Explore reproduced publication figures and benchmark tables.
- **Ablation & Fusion Lab**: Explore fuzzy Choquet integral non-additive fusion across 5 motor cortex electrode pairs with Shapley value attribution.

---

## 📁 Repository Structure

```
.
├── configs/
│   └── default_config.yaml         # Hyperparameters, electrode pairs, and training settings
├── src/
│   ├── data/
│   │   ├── loader.py               # Downloads & parses PhysioNet EEGMMIDB runs via MNE
│   │   ├── preprocessor.py         # Resampling, CAR, mu/beta bandpass, ICA, pair serialization
│   │   ├── synthetic.py            # Realistic synthetic EEG generator simulating ERD/ERS
│   │   └── dataset.py              # PyTorch Dataset and stratified 5:2:3 / 10-fold splitters
│   ├── models/
│   │   ├── minirocket_pipeline.py  # MiniRocket feature transform (PPV) + Ridge Classifier
│   │   ├── cnn_lstm.py             # Exact 13-layer PyTorch CNN-LSTM hybrid architecture
│   │   ├── ablations.py            # Hybrid CNN-GRU model, SVM, and Logistic Regression variants
│   │   └── fusion.py               # Choquet fuzzy integral non-additive multi-electrode fusion
│   ├── training/
│   │   ├── trainer_dl.py           # Deep learning trainer (Adam lr=1e-5, L2 decay=0.01)
│   │   └── trainer_rocket.py       # MiniRocket fit/predict with latency profiling
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, F1, Precision, Recall, Confusion Matrix, ROC-AUC
│   │   ├── profiler.py             # CPU latency (ms/sample & trial), parameters, peak RAM
│   │   └── plots.py                # Reproduces publication Figures 6, 7, 8, 9
│   └── app/
│       └── app.py                  # Interactive Streamlit dashboard
├── scripts/
│   ├── run_pipeline.py             # CLI pipeline runner
│   └── run_demo.py                 # Quick verification demo script
├── tests/
│   ├── test_preprocessing.py       # Unit tests for preprocessing & dataset splits
│   ├── test_minirocket.py          # Unit tests for MiniRocket pipeline
│   └── test_cnn_lstm.py            # Unit tests for CNN-LSTM & CNN-GRU models
├── results/                        # Generated figures and JSON benchmarks
├── pyproject.toml
└── README.md
```

---

## 🧪 Unit Testing

Run the full automated test suite:
```bash
python -m unittest discover -s tests
```

---

## 📜 Citation

```bibtex
@article{hwaidi2026motor,
  title={Motor imagery EEG signal classification using minimally random convolutional kernel transform and hybrid deep learning},
  author={Hwaidi, Jamal and Ghanem, Mohamed Chahine},
  journal={NeuroImage},
  volume={328},
  pages={121816},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2026.121816}
}
```
