# Facial Recognition using Olivetti Dataset

## Overview

This project implements classical machine learning approaches for facial recognition using the Olivetti Faces dataset. It demonstrates how to identify individuals from grayscale facial images by training classification models on dimensionally-reduced image features.

## Project Structure

The implementation performs the following tasks:

- **Data Handling**: Loads and processes the Olivetti facial image dataset
- **Preprocessing**: Normalizes and prepares image data for modeling
- **Dimensionality Reduction**: Applies PCA to extract essential facial features
- **Model Implementation**: Trains Gaussian Naive Bayes and Linear Discriminant Analysis classifiers
- **Performance Evaluation**: Analyzes model accuracy, precision, recall, and F1-score
- **Visualization**: Displays sample images and projection plots

## Dataset Information

The Olivetti Faces dataset contains:

- 400 grayscale facial images (64×64 pixels)
- 40 distinct subjects with 10 images per person
- Varied facial expressions, lighting conditions, and details

The dataset is included as:
- `olivetti_faces.npy`: Image data
- `olivetti_faces_target.npy`: Identity labels (0-39)

## Usage Instructions

1. Clone this repository
2. Run the Jupyter notebook: `Face_Recognition_on_Olivetti_Dataset.ipynb`
   - Compatible with both local Jupyter environments and Google Colab

## Results

Performance metrics on clean and noise-augmented test sets:

### Linear Discriminant Analysis (LDA)
| Test Set | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| Clean    | 0.93     | 0.95      | 0.93   | 0.93     |
| Noisy    | 0.92     | 0.94      | 0.92   | 0.91     |

### Gaussian Naive Bayes (NB)
| Test Set | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| Clean    | 0.88     | 0.92      | 0.88   | 0.87     |
| Noisy    | 0.80     | 0.87      | 0.80   | 0.80     |

The evaluation demonstrates that LDA consistently outperforms Naive Bayes, particularly when handling noisy data.

## Contributing

Contributions are welcome. Please feel free to fork the repository and submit pull requests with improvements or alternative model implementations.

## License

This project is available under the MIT License. See the LICENSE file for details.
