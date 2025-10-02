# Waste-Classification-CNN
A deep learning solution for automatic classification of waste images using Convolutional Neural Networks (CNN), leveraging TensorFlow/Keras and transfer learning through EfficientNetB0.

***

## Project Overview

This project trains and evaluates a CNN model to classify images of waste into six classes: cardboard, glass, metal, paper, plastic, and trash. It uses image augmentation and leverages transfer learning with EfficientNetB0 pretrained on ImageNet. The notebook includes training, evaluation, metrics reporting, and inference demonstrations.[1]

***

## Dataset

- The dataset should be organized in subfolders under `train`, `val`, and `test` for each class inside a root directory (default: `content/drive/MyDrive/Colab Notebooks/Garbage/`).
- Sample distribution in notebook:  
    - Train: 1766 images/6 classes
    - Validation: 377 images/6 classes
    - Test: 384 images/6 classes[1]
- Supported image formats: `.png`, `.jpg`, `.jpeg`.

***

## Installation

```bash
# Recommended environment: Google Colab with GPU, or a local setup with Python 3.x
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

***

## Usage Instructions

1. **Prepare Data:**  
   Organize your dataset as described above. Update the notebook path variables if needed.

2. **Model Training:**  
   - Run the notebook to mount Google Drive (for Colab) and set up dataset paths.
   - Adjust parameters like `IMG_SIZE` (default: 384x384) and `BATCH_SIZE` (default: 32).

3. **Transfer Learning:**  
   - EfficientNetB0 model is loaded and frozen for initial training.
   - Fine-tune specific layers after primary training.

4. **Training Callbacks:**  
   - Early stopping
   - Learning rate reduction upon plateau

5. **Evaluation:**  
   - Metrics include accuracy, loss, confusion matrix, and classification report (precision, recall, F1-score per class).
   - Example on test set: overall accuracy ~93%.[1]

6. **Model Saving/Loading:**  
   - Model is saved as `.keras` format after training.
   - To predict a single image, load model and preprocess image as shown in the notebook.[1]

***

## Example Results

- Cardboard: Precision 0.95, Recall 0.90
- Glass: Precision 0.95, Recall 0.95
- Metal: Precision 0.94, Recall 0.94
- Paper: Precision 0.90, Recall 0.93
- Plastic: Precision 0.96, Recall 0.95
- Trash: Precision 0.83, Recall 0.86
- Overall accuracy on test set: 0.93[1]

***

## File Structure

| Folder            | Contents                                      |
|-------------------|-----------------------------------------------|
| `train/`          | Training images subdivided by class           |
| `val/`            | Validation images subdivided by class         |
| `test/`           | Test images subdivided by class               |
| `.ipynb` notebook | Training and evaluation workflow              |
| Saved model files | `.keras` format after training                |

***

## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- numpy
- matplotlib
- OpenCV (for image preview/inference)
- Google Colab (recommended for fast GPU)

***

## How to Contribute

Feel free to fork the repo, open issues, and submit pull requests for enhancements, bug fixes, or new datasets.

***

## License

MIT License (recommended—adapt as needed).

***

## Acknowledgments

- EfficientNetB0 (TensorFlow/Keras applications)
- Dataset inspiration: TACO/TrashNet
- Contributors, tutorials, and open-source community

***

For further details, consult code comments and cell outputs in the notebook.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/118848277/6273b3d0-3876-424d-8174-896c3895189e/Waste_Classification_CNN.ipynb)
