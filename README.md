# Deep Learning Experiments: MLPs, CNNs, and Analog Clock-Time Prediction

TensorFlow/Keras experiments that compare dense and convolutional neural-network baselines across two controlled tasks:

1. **Image classification:** MLP and CNN models trained on Fashion-MNIST and CIFAR-10.
2. **Clock-time prediction:** a CNN that reads analog clock faces and predicts time using multiple target encodings under a circular error metric.

The objective is not to chase a leaderboard number. It is to examine how representation, model architecture, and label design affect generalisation.

---

## Project Snapshot

| Area              | Details                                                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**        | Complete academic deep-learning experiment suite                                                                                             |
| **Focus**         | Architecture comparison, controlled evaluation, reproducibility, and error analysis                                                          |
| **Framework**     | TensorFlow / Keras                                                                                                                           |
| **Tasks**         | Fashion-MNIST classification, CIFAR-10 classification, analog clock-time prediction                                                          |
| **Models**        | MLPs and CNNs                                                                                                                                |
| **Validation**    | Held-out test accuracy for classification; circular time error in minutes for clock prediction                                               |
| **Documentation** | Architecture notes in [`docs/architecture.md`](docs/architecture.md) and full report in [`deeplearning_report.pdf`](deeplearning_report.pdf) |
| **Scope**         | Educational research project; no transfer learning, augmentation pipeline, or production serving layer                                       |

---

## Why This Project Matters

The central question is:

> How does architecture choice affect performance when the visual structure of the problem changes?

A dense network can be adequate for simpler greyscale images. Convolutional models should be more useful when local spatial structure, colour variation, and visual complexity increase. The clock task adds a second question: how should a circular target such as time-of-day be encoded so that errors near the 12 o’clock boundary are handled sensibly?

---

## Results at a Glance

### Task 1 — Image Classification

| Model | Dataset       | Test Accuracy |
| ----- | ------------- | ------------: |
| MLP   | Fashion-MNIST |        88.47% |
| CNN   | Fashion-MNIST |        90.81% |
| CNN   | CIFAR-10      |        71.84% |

**Interpretation:** the CNN improves Fashion-MNIST performance over the MLP baseline and provides a meaningful baseline for the more visually complex CIFAR-10 dataset.

### Task 2 — Analog Clock-Time Prediction

Evaluation uses a circular time-distance metric, reported in minutes.

| Target encoding                   |    Mean error | Median error |
| --------------------------------- | ------------: | -----------: |
| Multi-head regression, unscaled   |    160.99 min |   154.00 min |
| **Multi-head regression, scaled** | **13.43 min** | **9.00 min** |
| Sine-cosine encoding              |    176.99 min |   175.00 min |
| 30-minute bucket classification   |    178.50 min |   179.00 min |
| 15-minute bucket classification   |    173.26 min |   172.00 min |

The scaled multi-head formulation performed best in this experiment. A later 150×150 input run underperformed, indicating a configuration or pipeline mismatch worth debugging rather than a valid claim that larger input resolution is worse.

---

## Repository Structure

```text
Image-Classification-Experiments-CNNs-MLPs-TensorFlow-Keras-/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── image_classification_experiments.py   # Root-level experiment script
├── A1_TASK_1.ipynb                       # Original Task 1 notebook
├── IDL_A1-Task_2.ipynb                   # Original Task 2 notebook
├── deeplearning_report.pdf               # Full written report
│
├── src/
│   ├── a1_task_1.py                      # Task 1: Fashion-MNIST / CIFAR-10 experiments
│   ├── idl_a1_task_2.py                  # Task 2: analog clock-time prediction
│   ├── A1_TASK_1.ipynb
│   ├── IDL_A1-Task_2.ipynb
│   └── IDL/                              # Supporting Task 2 resources
│
└── docs/
    └── architecture.md                   # Architecture and experiment notes
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/neeilnandal/Image-Classification-Experiments-CNNs-MLPs-TensorFlow-Keras-.git
cd Image-Classification-Experiments-CNNs-MLPs-TensorFlow-Keras-
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run the Experiments

### Task 1 — Image Classification

Run the refactored Python script:

```bash
python src/a1_task_1.py
```

Or open the original notebook:

```bash
jupyter notebook A1_TASK_1.ipynb
```

### Task 2 — Clock-Time Prediction

Run the Python implementation:

```bash
python src/idl_a1_task_2.py
```

Or open the original notebook:

```bash
jupyter notebook IDL_A1-Task_2.ipynb
```

> Dataset downloads are handled through the experiment code or the original notebook workflow. Training time depends on local hardware and TensorFlow configuration.

---

## Experimental Design

### Task 1: MLP versus CNN

The classification experiments compare:

* an MLP baseline to establish a non-convolutional reference point;
* CNN architectures that preserve local spatial structure;
* Fashion-MNIST and CIFAR-10 to test performance under different levels of visual complexity.

### Task 2: Representing Circular Time

The clock-time task compares several label encodings:

* unscaled multi-head regression;
* scaled multi-head regression;
* sine-cosine representation;
* coarse 30-minute bucket classification;
* finer 15-minute bucket classification.

The evaluation metric treats time as circular: an estimate near midnight should not be treated as maximally wrong when the true time is just after midnight.

---

## Key Findings

* CNNs outperformed the MLP baseline on Fashion-MNIST in this experiment.
* CIFAR-10 remained substantially harder than Fashion-MNIST, as expected from its colour variation and higher visual complexity.
* Label representation mattered more than expected for clock-time prediction.
* The scaled multi-head approach was the only tested encoding that produced low circular time error in this setup.
* A higher-resolution clock configuration did not transfer cleanly and should be treated as a debugging lead, not a completed improvement.

---

## Reproducibility Notes

* Use the versions pinned in `requirements.txt`.
* Results can vary slightly across TensorFlow versions, hardware, and random initialisation.
* Keep generated model weights, temporary datasets, and virtual environments out of version control unless required for a specific reproducibility release.
* The reported figures are experiment outputs from this repository, not universal benchmarks.

---

## Limitations

* No data augmentation.
* No pretrained or transfer-learning models.
* No repeated-seed confidence intervals.
* No confusion matrices or per-class precision/recall analysis.
* CIFAR-10 performance is sensitive to architecture, epoch count, hardware, and training configuration.
* The clock-time task requires further debugging for the higher-resolution configuration.

---

## Potential Extensions

* Add repeated-seed runs with mean and standard deviation reporting.
* Add confusion matrices and per-class performance analysis.
* Compare regularisation and augmentation strategies.
* Add transfer-learning baselines for CIFAR-10.
* Investigate the 150×150 clock-time configuration using saved intermediate predictions and data-pipeline checks.
* Add automated tests for preprocessing, label conversion, and circular-error calculation.

---

## Skills Demonstrated

`Python` · `TensorFlow` · `Keras` · `CNNs` · `MLPs` · `Computer Vision` · `Experiment Design` · `Model Evaluation` · `Hyperparameter Analysis` · `Reproducible ML`

---

## References

* Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O’Reilly, 2019.
* Abadi, M. et al. “TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.” 2015.
* Agostinelli, F. et al. “What Time Is It? Deep Learning Approaches for Circadian Rhythms.” *Bioinformatics*, 2016.

---

## License

Released under the [Apache-2.0 License](LICENSE).
