# Privacy-Preserving Federated Learning for Disease Diagnosis

This repository contains two final machine learning projects from my BSc Computer Science dissertation at Newcastle University (2025). The goal was to explore how **federated learning** can be used for disease diagnosis in a privacy-preserving manner, complying with GDPR principles while ensuring diagnostic accuracy.

## Project Overview

The project implements two separate models trained using TensorFlow Federated:

1. **Simple PPFL Model**  
   - **Task:** Stroke prediction from tabular symptom data  
   - **Model Type:** Dense neural network  
   - **Privacy Techniques:** Differential Privacy, SMOTE, data masking  
   - **Data:** Stroke prediction dataset (4981 records)

2. **Complex PPFL Model**  
   - **Task:** Pneumonia detection from chest X-rays  
   - **Model Type:** Convolutional neural network (CNN)  
   - **Privacy Techniques:** Differential Privacy, image downsampling, augmentation  
   - **Data:** Chest X-ray dataset (with preprocessing and simulated federated partitions)

Both models were trained using **federated averaging (FedAvg)**, and privacy was enhanced with techniques such as **data pseudonymisation**, **generalisation**, and **synthetic data generation**.

## Privacy Techniques Used

- **Differential Privacy**: Gaussian mechanism to add noise to model updates  
- **Synthetic Data**: SMOTE for class balancing in the stroke dataset  
- **Data Masking**: Sensitive fields obscured or generalised  
- **Pseudonymisation**: Replacement of identifiers with unlinkable keys  
- **Data Minimisation**: Only essential features retained for training

## Technologies

- `Python 3.10+`
- `TensorFlow 2.14`
- `TensorFlow Federated 0.84.0`
- `Scikit-Learn`
- `Pandas / NumPy`
- `Matplotlib`
- `imbalanced-learn` (SMOTE)
- `Google Colab` (used to simulate limited hardware conditions)

## Evaluation Highlights

| Model            | Accuracy | Precision | Recall | AUC  |
|------------------|----------|-----------|--------|------|
| **Tabular (PPFL)** | High     | High      | High   | ~0.85 |
| **Image (PPFL)**   | Moderate | Moderate  | Low    | ~0.60 |

- The tabular model performed well even under privacy constraints.
- The image model struggled with convergence, revealing trade-offs between complexity and noise-induced degradation.
