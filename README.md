# Deep Learning Assignment 1 — MLPs, CNNs, and Clock-Time Prediction

This repo contains two tasks built with TensorFlow/Keras:

1. **[Task 1 — Image Classification](task1_image_classification/)**: MLP and CNN models trained on Fashion-MNIST and CIFAR-10, exploring how architecture depth and hyperparameters affect accuracy.
2. **[Task 2 — Clock Time Prediction](task2_clock_time_prediction/)**: A CNN that reads an analog clock face and predicts the time, comparing five different label encodings (multi-head regression, sine–cosine, bucketed classification) under a custom circular error metric.

The full written report is in [`report/Assignment-1_Report.pdf`](report/Assignment-1_Report.pdf).

## Repo Structure

```
deep-learning-mlp-cnn/
├── task1_image_classification/   # MLP & CNN on Fashion-MNIST / CIFAR-10
├── task2_clock_time_prediction/  # Clock-time CNN with multiple label encodings
├── report/                       # Full PDF writeup
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd deep-learning-mlp-cnn
pip install -r requirements.txt
```

Each task folder contains both a Jupyter notebook (as originally run in Colab) and an equivalent `.py` script under `src/`.

## Results Summary

### Task 1 — Image Classification

| Model | Dataset | Test Accuracy |
|---|---|---|
| MLP | Fashion-MNIST | 88.47% |
| CNN | Fashion-MNIST | 90.81% |
| CNN | CIFAR-10 | 71.84% |

### Task 2 — Clock Time Prediction (75×75 images, validation error in minutes)

| Label Encoding | Mean Error (min) | Median Error (min) |
|---|---:|---:|
| Multi-Head (Unscaled) | 160.99 | 154.00 |
| **Multi-Head (Scaled)** | **13.43** | **9.00** |
| Sine–Cosine | 176.99 | 175.00 |
| Bucket 30 min | 178.50 | 179.00 |
| Bucket 15 min | 173.26 | 172.00 |

The best-performing encoding (multi-head, scaled minutes) was scaled up to 150×150 inputs but underperformed (mean 179.39 min, median 179.00 min), suggesting a configuration mismatch flagged for future debugging — details in the [Task 2 README](task2_clock_time_prediction/README.md#known-issue--150150-model).

OODA Summary
Observe
Fashion-MNIST and CIFAR-10 have different visual complexity. A dense network may work on simple images but struggle with richer color images.
Orient
The right comparison is not only accuracy. It is the relationship between dataset structure, model architecture, and generalization.
Decide
Use MLPs as baselines, CNNs as structure-aware models, and a width sweep to test dense-network capacity.
Act
Refactor the notebook into a reproducible script that trains models, saves metrics, plots curves, and produces comparison-ready outputs.
Founder-Style Diagnosis
User
Students, recruiters, ML reviewers, or hiring managers checking whether the project demonstrates practical ML understanding.
Pain Point
A raw notebook with scattered outputs does not clearly show what was learned.
Better Product
A clean experiment suite with repeatable runs, corrected labels, saved results, and a clear README.
Smallest Useful Version
One script that trains Fashion-MNIST and CIFAR-10 baselines and saves a result summary.
Current Version
The current version includes clean model builders, consistent preprocessing, a real CNN baseline, experiment tracking, plots, and CLI controls.
Security and Reproducibility Notes
This project is low-risk.
Area	Status
Secrets	None
API keys	None
User data	None
External datasets	Downloaded through Keras dataset utilities
Code execution risk	Standard local Python training script
Privacy risk	Low
Reproducibility	Random seeds added
Recommended safeguards:
·	Do not commit large generated model files unless needed
·	Keep virtual environments out of Git
·	Save only compact result files and plots
·	Document TensorFlow version in `run\_metadata.json`
Scientific Skills Demonstrated
This project shows:
·	Experimental comparison
·	Baseline modeling
·	Model architecture reasoning
·	Hyperparameter sensitivity analysis
·	Train/test evaluation
·	Learning-curve interpretation
·	Reproducibility controls
·	Result logging
The scientific value is not in reaching maximum accuracy. It is in asking a controlled question:
> How does architecture choice affect image classification across datasets of different visual complexity?
Limitations
·	No data augmentation
·	No transfer learning
·	No confusion matrix yet
·	No per-class precision or recall
·	No statistical repeated runs
·	No model checkpoint comparison
·	CIFAR-10 results depend heavily on training duration and hardware
SEO Keywords
Relevant keywords:
·	Fashion-MNIST classification
·	CIFAR-10 classification
·	TensorFlow Keras CNN
·	MLP vs CNN comparison
·	image classification experiment
·	deep learning portfolio project
·	neural network baseline
·	convolutional neural network
·	Keras image classification
·	hyperparameter sweep
Repository Topics
```text
tensorflow
keras
deep-learning
cnn
mlp
image-classification
fashion-mnist
cifar10
neural-networks
computer-vision
python
```


## References

- A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd Edition, O'Reilly, 2019.
- M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," 2015.
- F. Agostinelli et al., "What time is it? Deep learning approaches for circadian rhythms," *Bioinformatics*, vol. 32, no. 12, pp. i8–i17, 2016.




