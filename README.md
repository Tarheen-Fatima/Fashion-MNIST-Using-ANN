# Binary Classification using CNN on Fashion MNIST Dataset

This repository contains a Convolutional Neural Network (CNN) model implementation for binary classification on the Fashion MNIST dataset. The task is to classify between two classes of clothing items (T-shirts/tops and trousers) using a neural network implemented with TensorFlow and Keras.

## Dataset

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) consists of 28x28 grayscale images of 10 different types of clothing items. For this project, we are only using two categories:

- Class 0: T-shirt/top
- Class 1: Trouser

The dataset is automatically loaded using `tensorflow.keras.datasets`.

## Model Architecture

The model architecture is a simple CNN designed for binary classification. Here's an overview:

1. **Input Layer**: The input is a 28x28 grayscale image, reshaped to (28, 28, 1).
2. **Convolutional Layer**: A 2D convolution layer with 32 filters of size (3x3), followed by ReLU activation.
3. **MaxPooling Layer**: A max pooling layer with a pool size of (2x2).
4. **Flatten Layer**: Flattening the 2D matrix into a 1D vector.
5. **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation.
6. **Output Layer**: A fully connected layer with 1 neuron and sigmoid activation for binary classification.

## Model Summary

- **Loss function**: Binary cross-entropy
- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.01
- **Evaluation metric**: Accuracy

## Training and Evaluation

The model is trained for 10 epochs using a batch size of 32. After training, the model is evaluated on the test set.

```bash
Test accuracy: 0.9876
```

## Customization

- **Changing classes**: The code is designed to filter the dataset for two classes (T-shirt/top and trouser). You can easily modify the filtering conditions to include other classes from the Fashion MNIST dataset by changing the conditions for `y_train` and `y_test`.
- **Changing architecture**: Feel free to modify the model architecture by adding/removing layers or adjusting hyperparameters like the number of epochs, batch size, or optimizer.
