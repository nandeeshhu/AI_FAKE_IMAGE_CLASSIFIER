# AI Fake Image Classifier

This repository contains code and resources for a deep learning model designed to classify images as either real or fake. The model uses the EfficientNet architecture and is trained on a dataset of real and fake images.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Performance](#performance)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The AI Fake Image Classifier is a binary classification model that distinguishes between real and AI-generated (fake) images. This project aims to provide a robust solution for detecting AI-generated content, which is increasingly prevalent in various media.

## Dataset

### Real Images
- **Source:** A collection of real images from various public datasets.
- **Number of Images:** 46,998

### Fake Images
- **Source:** AI-generated images using various generative models.
- **Number of Images:** 47,000

The dataset is stored in Google Drive and consists of two main directories:
- `Real`: Contains real images.
- `Fake`: Contains AI-generated images.

## Model Architecture

The model is based on the EfficientNet-B0 architecture, a state-of-the-art convolutional neural network known for its efficiency and accuracy. The classifier layer has been modified to output two classes (real and fake).

## Training and Evaluation

The dataset is split into three parts:
- **Training Set:** 91,998 images
- **Validation Set:** 1,000 images
- **Test Set:** 1,000 images

### Data Augmentation and Preprocessing
- Images are resized to 224x224 pixels.
- Normalization is applied using the mean and standard deviation of the ImageNet dataset.

### Training
- **Learning Rate:** 0.01
- **Batch Size:** 8
- **Epochs:** 15

### Validation and Testing
- The model's performance is evaluated on the validation set after each epoch.
- The final evaluation is conducted on the test set.

## Performance

### Validation Performance
- **Best Validation Accuracy:** 98.5%
- **Validation Loss:** 0.05

### Test Performance
- **Test Accuracy:** 98.3%


## Usage

### Prerequisites
- Python 3.8 or higher
- PyTorch
- torchvision
- PIL
- numpy

### Installation

1. Clone this repository to your local machine, specifying the branch containing the project files:

    ```bash
    git clone -b my-new-branch https://github.com/nandeeshhu/AI_FAKE_IMAGE_CLASSIFIER.git
    cd your-repository
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
