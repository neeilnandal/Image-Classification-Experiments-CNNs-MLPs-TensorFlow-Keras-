Image Classification Experiments with MLPs and CNNs
A clean TensorFlow/Keras experiment suite comparing MLP and CNN baselines on Fashion-MNIST and CIFAR-10.
This project started as a notebook-style academic experiment. It has been refactored into a reproducible Python script with consistent preprocessing, corrected model labels, proper CNN baselines, saved results, training-curve plots, and a small hyperparameter sensitivity experiment.
Repository Summary
Field	Details
Project type	Deep learning image classification experiment
Datasets	Fashion-MNIST, CIFAR-10
Models	Dense MLP baselines, CNN baselines
Framework	TensorFlow / Keras
Language	Python
Outputs	Accuracy/loss curves, experiment summary CSV, neuron sweep CSV
Main learning goal	Compare how architecture choice affects image classification performance
Problem
Image datasets do not all behave the same way.
A dense MLP can work reasonably well on simple grayscale images, but it ignores spatial structure. A CNN is designed to learn local visual patterns such as edges, shapes, textures, and object parts.
The core question behind this project is:
> How do dense MLPs and CNNs behave differently on simple image data versus harder natural-image data?
Fashion-MNIST and CIFAR-10 make a useful comparison:
Dataset	Image Type	Difficulty
Fashion-MNIST	28×28 grayscale clothing images	Easier
CIFAR-10	32×32 color object images	Harder
First-Principles View
The real goal is not “train a neural network.”
The real goal is:
> Understand when a simple dense network is enough, and when image structure makes convolution useful.
Flattening an image turns pixels into a long vector. That can work for simple data, but it discards spatial relationships. CNNs preserve local structure, which makes them more suitable for natural images.
This project demonstrates that distinction through controlled baseline experiments.
Tech Stack
Area	Technology
Deep Learning	TensorFlow, Keras
Data Handling	NumPy, Pandas
Visualization	Matplotlib
Datasets	Keras datasets
Experiment Tracking	CSV outputs
Reproducibility	Fixed random seeds
Project Structure
```text
image-classification-experiments/
│
├── image\_classification\_experiments.py
├── README.md
├── requirements.txt
├── .gitignore
└── results/
    ├── experiment\_summary.csv
    ├── neuron\_sweep\_results.csv
    ├── run\_metadata.json
    └── training\_curves/
        ├── fashion-mnist\_mlp\_baseline\_accuracy.png
        ├── fashion-mnist\_mlp\_baseline\_loss.png
        ├── fashion-mnist\_cnn\_baseline\_accuracy.png
        ├── fashion-mnist\_cnn\_baseline\_loss.png
        ├── cifar-10\_mlp\_baseline\_accuracy.png
        ├── cifar-10\_mlp\_baseline\_loss.png
        ├── cifar-10\_cnn\_baseline\_accuracy.png
        ├── cifar-10\_cnn\_baseline\_loss.png
        └── cifar10\_neuron\_sweep.png
```
The `results/` directory is generated when the script runs.
Installation
Clone the repository:
```bash
git clone https://github.com/your-username/image-classification-experiments.git
cd image-classification-experiments
```
Create a virtual environment:
```bash
python -m venv venv
```
Activate it.
On macOS or Linux:
```bash
source venv/bin/activate
```
On Windows:
```bash
venv\\Scripts\\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Requirements
```text
tensorflow
numpy
pandas
matplotlib
```
Run the Experiments
Run all experiments:
```bash
python image\_classification\_experiments.py
```
Run faster demo settings:
```bash
python image\_classification\_experiments.py --epochs-fashion 5 --epochs-cifar 5 --epochs-sweep 2
```
Experiments
1. Fashion-MNIST MLP Baseline
A dense baseline using flattened 28×28 grayscale images.
Architecture:
```text
Input 28×28×1
Flatten
Dense 300 ReLU
Dense 100 ReLU
Dense 10 Softmax
```
Purpose:
> Test how far a simple dense network can go on a clean grayscale image dataset.
2. Fashion-MNIST CNN Baseline
A real convolutional baseline.
Architecture:
```text
Input 28×28×1
Conv2D 32
MaxPooling2D
Conv2D 64
MaxPooling2D
Flatten
Dropout
Dense 128 ReLU
Dropout
Dense 10 Softmax
```
Purpose:
> Compare a structure-aware image model against a dense baseline.
3. CIFAR-10 MLP Baseline
A dense baseline using flattened 32×32×3 color images.
Architecture:
```text
Input 32×32×3
Flatten
Dense 512 ReLU
Dropout
Dense 512 ReLU
Dropout
Dense 10 Softmax
```
Purpose:
> Show the limitation of dense networks on harder color-image data.
4. CIFAR-10 CNN Baseline
An improved CNN with convolution blocks, batch normalization, pooling, and dropout.
Purpose:
> Build a stronger baseline for natural-image classification while keeping the model simple enough for a portfolio project.
5. CIFAR-10 MLP Width Sweep
A sensitivity experiment testing hidden-layer widths:
```text
10, 20, 30, 40, 50, 75, 100
```
Purpose:
> Measure whether increasing dense-layer width improves CIFAR-10 MLP performance.
Outputs
After running, the script saves:
```text
results/experiment\_summary.csv
results/neuron\_sweep\_results.csv
results/run\_metadata.json
results/training\_curves/\*.png
```
Experiment Summary
`experiment\_summary.csv` contains:
Column	Meaning
experiment_name	Dataset and architecture
dataset	Fashion-MNIST or CIFAR-10
architecture	MLP or CNN
epochs	Number of completed epochs
batch_size	Training batch size
optimizer	Optimizer used
learning_rate	Learning rate
train_accuracy	Final training accuracy
validation_accuracy	Final validation accuracy
test_accuracy	Test accuracy
test_loss	Test loss
parameter_count	Number of trainable parameters
notes	Short interpretation
Original Baseline Results
The original rough experiment produced these approximate results:
Dataset	Claimed Model	Actual Model	Test Accuracy
Fashion-MNIST	MLP	Dense MLP	89.20%
Fashion-MNIST	CNN	Dense MLP with dropout	88.02%
CIFAR-10	MLP	Dense MLP	38.38%
CIFAR-10	CNN	Conv2D CNN	39.51%
CIFAR-10	Neuron sweep	Dense MLP sweep	Best: 42.67%
Expected Interpretation
The expected pattern is:
Comparison	Expected Insight
Fashion-MNIST MLP vs CNN	Both can perform well, CNN should usually generalize better
CIFAR-10 MLP vs CNN	CNN should outperform MLP because CIFAR-10 has spatial and color structure
Width sweep	More neurons can help MLP performance, but architecture matters more than width alone
Data Analysis Notes
This project demonstrates a basic experimental ML workflow:
1.	Load benchmark datasets
2.	Normalize pixel values
3.	Define architecture baselines
4.	Train models under consistent settings
5.	Evaluate test performance
6.	Save metrics
7.	Plot learning curves
8.	Compare architecture behavior
The results should not be treated as state-of-the-art. They are controlled baselines for learning and comparison.
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
Portfolio One-Liner
Refactored a Keras image-classification experiment into a reproducible pipeline comparing MLP and CNN baselines on Fashion-MNIST and CIFAR-10, with saved metrics, training curves, and a neuron-width sensitivity study.


