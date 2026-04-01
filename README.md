MNIST Digit Classification using PyTorch

Project Overview:
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model is trained on grayscale images of size 28×28 and predicts digits from 0 to 9.

Objectives:
Develop a CNN-based image classification model
Understand the complete deep learning pipeline: data loading, training, evaluation, and inference
Gain practical experience with PyTorch components such as nn.Module, optimizers, and loss functions
Implement GPU acceleration for efficient training

Dataset:
The MNIST dataset consists of-
60,000 training images
10,000 testing images
Grayscale images of size 28×28
10 classes representing digits from 0 to 9
In this project, the dataset is loaded from local .idx files.

Project Structure:
mnist-classifier/
│
├── model.py              # CNN architecture
├── train.py              # Training script
├── test.py               # Inference and visualization
│
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
├── t10k-labels.idx1-ubyte
│
└── Models/
    └── model.pt          # Saved trained model (ignored in Git)

Requirements:
Install the required dependencies-
pip install torch torchvision numpy tqdm opencv-python torchsummary

Model Architecture:

The model consists of-
Two convolutional layers with ReLU activation and max pooling
Dropout layer for regularization
Two fully connected layers
Output layer with 10 classes

Training:
To train the model, run-
python train.py

The training process includes-
Forward propagation
Loss computation using CrossEntropyLoss
Backpropagation
Weight updates using the Adam optimizer

After training, the model is saved as-
Models/model.pt

Testing and Inference:

To test the model and visualize predictions-
python test.py

This will-
Load the trained model
Perform inference on test images
Display each image using OpenCV
Show the predicted digit as the window title

GPU Support:

The code automatically detects and uses a GPU if available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Conclusion:
This project demonstrates a complete deep learning workflow for image classification using PyTorch, including model design, training, evaluation, and real-time inference visualization.
