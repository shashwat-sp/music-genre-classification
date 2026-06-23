# Music Genre Classification

## Introduction

Music Genre Classification is a prominent task in **Music Information Retrieval (MIR)** that aims to automatically categorize songs into predefined genres based on their acoustic characteristics. With the rapid growth of digital music libraries and streaming platforms, automated genre classification has become increasingly important for music organization, recommendation systems, and content retrieval.

This project develops a music genre classification system using both **Machine Learning (ML)** and **Deep Learning (DL)** approaches. Various audio features are extracted from music files and used to train and evaluate multiple classification models, allowing comparison between traditional ML techniques and deep learning architectures.

---

## Problem Statement

Manual categorization of music is time-consuming and subjective. An automated system capable of accurately identifying genres based on audio content can significantly enhance music organization, recommendation engines, and user experience.

---

## Objectives

* Develop an automated music genre classification system.
* Perform preprocessing and analysis of audio data.
* Extract meaningful features from audio signals.
* Implement traditional Machine Learning algorithms.
* Implement Deep Learning models for improved performance.
* Compare different approaches and models.
* Evaluate performance using standard classification metrics.

---

## Dataset

### GTZAN Genre Collection Dataset

| Attribute               | Value      |
| ----------------------- | ---------- |
| Total Audio Files       | 1000       |
| Genres                  | 10         |
| Samples per Genre       | 100        |
| Audio Format            | WAV        |
| Duration per Audio File | 30 seconds |

### Genres Included

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

---

## System Architecture

```text
Audio Dataset
      ↓
Data Preprocessing
      ↓
Feature Extraction
(MFCC, Chroma, Spectral Features)
      ↓
Feature Scaling
      ↓
Machine Learning Models
(SVM, Random Forest, KNN)
      ↓
Deep Learning Models
(CNN)
      ↓
Genre Prediction
      ↓
Performance Evaluation
```

---

## Methodology

### 1. Data Collection

The GTZAN dataset is used as the primary dataset and serves as a benchmark for music genre classification tasks.

### 2. Data Preprocessing

* Loading audio files using **Librosa**
* Noise reduction and normalization
* Audio segmentation
* Train-test splitting

### 3. Feature Extraction

#### Mel Frequency Cepstral Coefficients (MFCC)

Captures timbral properties of audio and is widely used in speech and audio classification.

#### Chroma Features

Represent harmonic and pitch information.

#### Spectral Centroid

Measures the brightness of sound.

#### Spectral Rolloff

Indicates the frequency below which most spectral energy is concentrated.

#### Zero Crossing Rate (ZCR)

Measures the rate at which the signal changes sign.

#### Root Mean Square Energy (RMS)

Represents the energy level of audio signals.

---

## Machine Learning Models

### Support Vector Machine (SVM)

* Effective for high-dimensional feature spaces.
* Suitable for classification problems.

### Random Forest

* Ensemble learning approach based on decision trees.
* Reduces overfitting and improves robustness.

### K-Nearest Neighbors (KNN)

* Instance-based learning algorithm.
* Classifies samples using neighboring data points.

---

## Deep Learning Model

### Convolutional Neural Network (CNN)

CNN learns discriminative patterns from spectrograms or extracted features.

#### Architecture

```text
Input Layer
     ↓
Convolution Layer
     ↓
ReLU Activation
     ↓
Max Pooling Layer
     ↓
Flatten Layer
     ↓
Fully Connected Layer
     ↓
Softmax Output Layer
```

---

## Technologies and Tools

### Programming Language

* Python

### Libraries and Frameworks

* NumPy
* Pandas
* Librosa
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow
* Keras

### Development Environment

* Jupyter Notebook
* Google Colab
* Visual Studio Code

---

## Workflow

```text
Collect Dataset
      ↓
Preprocess Audio Data
      ↓
Feature Extraction
      ↓
Train-Test Split
      ↓
Train ML Models
      ↓
Train Deep Learning Models
      ↓
Performance Evaluation
      ↓
Compare Models
      ↓
Predict Genres for Unseen Songs
```

---

## Evaluation Metrics

The performance of models is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

---

## Expected Outcomes

* Automatic classification of songs into genres.
* Analysis of audio feature extraction techniques.
* Comparison of Machine Learning and Deep Learning approaches.
* Insights into strengths and limitations of different models.
* Potential applications in recommendation systems and Music Information Retrieval.

---

## Advantages

* Reduces manual effort in music categorization.
* Efficient organization of large music collections.
* Supports recommendation systems.
* Enhances music retrieval and accessibility.
* Provides a foundation for intelligent audio analysis.

---

## Limitations

* Similar genres may have overlapping characteristics.
* Performance depends heavily on dataset quality.
* Deep learning models require significant computational resources.
* Noise and recording quality may affect classification accuracy.

---

## Future Scope

* Support larger and more diverse datasets.
* Real-time music genre prediction.
* Deployment as a web application.
* Integration with music streaming platforms.
* Exploration of advanced architectures such as LSTM, CRNN, and Transformers.

---

## Conclusion

This project presents an automated **Music Genre Classification System** using both Machine Learning and Deep Learning techniques. By extracting meaningful audio features and evaluating multiple models, the system investigates different approaches to genre prediction and contributes to the field of **Music Information Retrieval (MIR)**.

---
