# CIFAR-10 Image Classification Using Deep Learning

## 1. Introduction
Deep Learning is a branch of machine learning that uses neural networks with multiple layers to learn patterns and representations from data. In computer vision, Convolutional Neural Networks (CNNs) are widely used to extract visual features and perform image classification.

This project explores the development of a CNN-based image classification model using PyTorch and the CIFAR-10 dataset. The goal is to understand the complete deep learning workflow, from data preparation and model design to training, evaluation, and performance analysis.

Rather than focusing only on achieving high accuracy, this project emphasizes understanding how training configurations affect model performance, how to identify overfitting and underfitting, and how to make evidence-based improvements.

---

## 2. Project Objectives
The main objectives of this project are:
* Understand the fundamental concepts of deep learning and convolutional neural networks.
* Build an image classification model using PyTorch.
* Train the model using a selected loss function and optimization algorithm.
* Monitor training and validation performance throughout the learning process.
* Evaluate the final model using a separate test dataset.
* Analyze model limitations and identify opportunities for improvement.

---

## 3. Problem Statement
Image classification is a computer vision task in which a model assigns an input image to one of several predefined categories.

In this project, the model receives a color image and predicts one of ten CIFAR-10 classes: 
> Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

The task is challenging because images have low resolution, and objects within the same class can have different appearances, backgrounds, and positions. The model must learn visual patterns that generalize to images it has not encountered during training.

---

## 4. Dataset
The CIFAR-10 dataset contains 60,000 color images of size 32 × 32 pixels across 10 classes.

The standard dataset consists of 50,000 training images and 10,000 test images. In this project, the original training set is split into training and validation subsets. The training and validation subsets are created using stratified sampling to preserve the class distribution.

---

## 5. Methodology

### 5.1 Data Preprocessing
Images are converted into tensors and normalized using the channel-wise mean and standard deviation of CIFAR-10. Normalization helps place input values on a more consistent scale, which can support stable model optimization.

### 5.2 Data Augmentation
The training pipeline uses random cropping and horizontal flipping to generate varied versions of the training images. Data augmentation helps the model learn features that are less sensitive to small changes in object position and orientation. 

*Note: Augmentation is applied only to training images. Validation and test images use consistent preprocessing without random augmentation.*

### 5.3 CNN Architecture
The model uses convolutional layers to extract visual features, batch normalization to help stabilize training, ReLU activations to introduce non-linearity, and max pooling to reduce spatial dimensions. Dropout is used to help reduce overfitting. Adaptive average pooling and fully connected layers produce class scores for the ten categories.

### 5.4 Training Configuration
The initial experiment uses the following configuration:

| Parameter | Initial Setting |
| :--- | :--- |
| **Framework** | PyTorch |
| **Model** | Custom CNN |
| **Loss function** | Cross-Entropy Loss |
| **Label smoothing** | 0.05 |
| **Optimizer** | AdamW |
| **Initial learning rate** | 0.002 |
| **Weight decay** | 0.001 |
| **Batch size** | To be documented from the training script |
| **Number of epochs** | 100 |
| **Learning-rate scheduler** | CosineAnnealingLR |
| **Scheduler T_max** | 20 |
| **Training augmentation** | Random crop and horizontal flip |

These values define the baseline experiment. Further experiments can change one or more settings to study their impact on model performance.

---

## 6. Evaluation Metrics
The model is evaluated using multiple metrics rather than accuracy alone:
* **Training loss:** Measures prediction error on the training data.
* **Validation loss:** Measures prediction error on the validation data.
* **Training accuracy:** Percentage of training examples classified correctly.
* **Validation accuracy:** Percentage of validation examples classified correctly.
* **Test accuracy:** Percentage of test examples classified correctly after model selection.
* **Confusion matrix:** Shows which classes are correctly classified and which classes are confused with one another.

Loss and accuracy curves are used to examine learning behavior and detect possible overfitting or underfitting.

---

## 7. Results and Analysis
*Actual results will be added after the training experiment is completed.*

Planned visualizations include:
* Training and validation loss curves.
* Training and validation accuracy curves.
* Test-set confusion matrix.
* Example predictions with correct and incorrect classifications.

The analysis will examine whether the model converges, whether a gap develops between training and validation performance, and which classes are most difficult to classify.

---

## 8. Conclusion
This project provides a practical introduction to the end-to-end deep learning workflow for image classification.

The final conclusion will be based on measured validation and test results, learning curves, and class-level performance. The analysis will identify the model's strengths, limitations, and potential improvements.

<img width="1338" height="465" alt="image" src="https://github.com/user-attachments/assets/604d36a9-b24e-46a2-be8e-48283365e0f8" />


---

## 9. Technologies Used
* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn

