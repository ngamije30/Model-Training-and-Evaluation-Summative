# Chest X-Ray Pneumonia Detection

## Project Overview
This project implements a machine learning pipeline to classify chest X-ray images as either normal or indicative of pneumonia. The project compares traditional machine learning approaches using scikit-learn with deep learning approaches using TensorFlow.

## Problem Statement
Pneumonia is a leading cause of death worldwide, particularly in children under 5 years old. Early and accurate detection of pneumonia through chest X-ray analysis can significantly improve patient outcomes and reduce healthcare burdens. This project aims to develop an automated system for pneumonia detection using machine learning techniques.

## Dataset
- **Source**: Chest X-Ray Images (Pneumonia) dataset from Kaggle
- **Size**: 5,856 chest X-ray images
- **Classes**: Normal (1,583 images) and Pneumonia (4,273 images)
- **Patient Age**: 1-5 years (pediatric patients)
- **Image Format**: JPG files in grayscale

## Methodology
1. **Traditional ML**: Feature extraction using HOG, PCA, and other techniques followed by classification using SVM, Random Forest, and Logistic Regression
2. **Deep Learning**: CNN architectures including custom models and transfer learning with pre-trained models (VGG16, ResNet50, MobileNet)

## Project Structure
```
chest-xray-pneumonia-detection/
├── dataset/                    # Dataset storage
├── notebook/m                  # Jupyter notebooks
├── READEME.md
```

## Getting Started
1. Clone this repository
2. Install dependencies
3. Download the dataset from Kaggle
4. Run the main notebook: `notebooks/NGAMIJE_Davy_Summative Assignment - Model Training and Evaluation_Pneumonia_detection.ipynb`

## Results
[Results will be shown in the notebook]

## References
1. Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"
 2. Kermany, D.S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" 
3. World Health Organization (2021). "Pneumonia Fact Sheet" 4. Franquet, T. (2018). "Imaging of Community-acquired Pneumonia" 
5. Wang, X., et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database" 
6. Dalal, N., & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection" 
7. Ojala, T., et al. (2002). "Multiresolution Gray-Scale and Rotation Invariant Texture Classification" 
8. Liang, G., et al. (2020). "Evaluation of Traditional Machine Learning Methods for Medical Imaging Analysis" 
9. Breiman, L. (2001). "Random Forests" 
10. Zhang, Y., et al. (2018). "Feature Engineering for Medical Image Analysis" 
11. He, K., et al. (2016). "Deep Residual Learning for Image Recognition" 
12. Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" 
13. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

