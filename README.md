# BAN6420-Module-6-Assignment-Fashion-MNIST-Classification

# Fashion MNIST CNN Project

# Overview
## This project focuses on building a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset using both Python and R. The Fashion MNIST dataset consists of 70,000 grayscale images of 28x28 pixels, representing 10 different categories of clothing items. The goal is to accurately classify these images into their respective categories using deep learning models.

## The project is divided into two parts:

- Python Implementation: We use the Keras API (running on top of TensorFlow) to build a CNN that includes several convolutional layers, max-pooling layers, a dense fully connected layer, and a softmax output layer to classify the 10 different fashion items.

- R Implementation: Similarly, in R, we implement the CNN using the tensorflow and keras libraries to replicate the model architecture built in Python. This allows us to work with the same dataset in two different environments and compare model performances if desired.

## Requirements
- Python
- `Python 3.x` (preferably 3.8 or above)
- `TensorFlow`
- `Keras`
- `NumPy`
- `Matplotlib` (for visualizing the results)

- R
- R version 4.x` or higher
- `tensorflow` R package
- keras R package
- reticulate for Python-R integration
- `ggplot2` (for visualizations in R)

## You can install the necessary Python dependencies using the following command:
pip install `-r requirements.txt`


## You can install the necessary R packages using:
install.packages(c(`tensorflow`, `keras`, `reticulateV`, `ggplot2`))


## Make sure to initialize the tensorflow and keras packages in R by running:
library(tensorflow)
install_tensorflow()

## Model Architecture
The Convolutional Neural Network (CNN) model architecture used in both Python and R includes the following layers:

Input Layer: 28x28 grayscale images, which are reshaped into (28, 28, 1) to be compatible with the Keras CNN model. Convolution Layer 1: 32 filters, 3x3 kernel, ReLU activation. Max Pooling Layer 1: 2x2 pooling to reduce dimensionality. Convolution Layer 2: 64 filters, 3x3 kernel, ReLU activation. Max Pooling Layer 2: 2x2 pooling. Convolution Layer 3: 128 filters, 3x3 kernel, ReLU activation. Flattening Layer: Converts the 2D feature maps into a 1D vector. Dense Fully Connected Layer: 128 units, ReLU activation. Output Layer: 10 units (one for each clothing category), softmax activation for classification.

## Training the Model
The model is trained on 60,000 images from the Fashion MNIST dataset, with 10,000 images set aside for testing. The training is done using the Adam optimizer and categorical crossentropy as the loss function.

Key Hyperparameters
Batch Size: 32
Epochs: 10
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy

## Predictions
After training the CNN model, predictions are made on two sample images from the test set. The predictions are compared with the actual labels, and the classification results are displayed.

## Visualization
In both Python and R, we use visualization libraries to plot the sample images along with their predicted labels to visually verify the results.

## How to Run
*Python *
Clone the repository or unzip the project folder.
Run the Python script
- python `fashion_mnist_cnn.ipynb`

*R*
Ensure that you have the necessary R libraries installed.
Run the following commands in R
- `fashion_mnist_cnn.R`

## Conclusion
This project demonstrates how to build a CNN using both Python and R for image classification tasks. By comparing the two implementations, we gain insight into the flexibility and power of the TensorFlow and Keras libraries across different programming environments.



## Author
Developed by Calistus Chukwuebuka Ndubuisi.
