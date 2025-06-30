# Distracted Driver Detection
## 1. Introduction
This repository implements a deep-learning pipeline for distracted-driver classification, categorizing in-vehicle images into one of ten predefined behavior classes. Built with TensorFlow and Keras, the model ingests the State Farm Distracted Driver dataset, applies on-the-fly preprocessing, and learns robust feature representations via a customizable convolutional neural network architecture

## 2. Overview

This project implements a memory-efficient TensorFlow/Keras pipeline to classify in-vehicle images into one of ten distracted-driver behaviors. Key features include:

- **Data Pipeline**  
  - Stratified 75/10/15 train/validation/test splits  
  - On-the-fly pixel normalization  
  - Optimized caching & prefetching via `tf.data`

- **Model Architecture**  
  - Parameterized 4-layer CNN builder  
  - Alternating ReLU & ELU activations  
  - Dropout regularization & softmax output  
  - Trained with Adam optimizer & categorical crossentropy

- **Hyperparameter Optimization**  
  - Sweeps over filter depths, kernel sizes & learning rates  
  - Achieves ~99 % accuracy on validation & held-out test sets

- **Evaluation & Analysis**  
  - Confusion-matrix generation  
  - Top-3 prediction visualization for detailed error analysis

## 3. Project Structure
The implementation is organized into the following stages:
- **Library Imports:** Import TensorFlow, NumPy, Matplotlib, and supporting utilities for data I/O and visualization.
- **Dataset Preparation:** Download and extract the State Farm dataset; organize images into class-labeled directories.
- **Data Loading & Preprocessing:** Construct a `tf.data` pipeline that resizes frames, performs pixel-wise normalization, and applies stratified 75/10/15 train/validation/test splits. Optimized caching and prefetching ensure efficient throughput.
- **Model Definition:** Instantiate a parameterized CNN builder that stacks four convolutional blocks with configurable filter depths, kernel sizes, ReLU/ELU activations, and dropout layers culminating in a softmax classification layer.
- **Optimization Setup:** Configure the Adam optimizer with categorical crossentropy loss; define learning-rate schedules for systematic hyperparameter exploration.
- **Training Loop:** Execute multi-epoch training while logging accuracy and loss on both training and validation subsets.
- **Evaluation & Analysis:** Generate confusion matrices and top-three prediction visualizations to assess per-class performance and guide error analysis.
  
## 4. How to Run the Project
- **Prerequisites:** Python 3.8+, TensorFlow 2.x, NumPy, Matplotlib, scikit-learn.
- **Configuration:** Clone the repository and install dependencies via `pip install -r requirements.txt`.
- **Execution:** Launch the main training script:
```bash
python train_distracted_driver.py \
  --data_dir /path/to/statefarm_dataset \
  --batch_size 32 \
  --epochs 20
```
Adjust flags to modify splits, network depth, or learning-rate schedules as needed.

## 5. Dataset Availability
- State Farm Distracted Driver Dataset: Images and labels are hosted at Kaggle:
https://www.kaggle.com/c/state-farm-distracted-driver-detection 
