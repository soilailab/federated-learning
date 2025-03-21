"""
Defines the CNN model architecture and compilation logic.
References hyperparameters and settings from config.py.
"""
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD

from config import config

seed = config["seed"]
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
tf.keras.utils.set_random_seed(seed)

def create_model():
    """
    Creates and compiles a Convolutional Neural Network (CNN) model based on hyperparameters
    defined in the configuration file (`config.py`).
    
    The model consists of two convolutional layers followed by fully connected layers.
    The model is designed for regression tasks with Mean Squared Error (MSE) loss.
    
    Returns:
        model (tensorflow.keras.Model): Compiled CNN model ready for training.
    """
    filters1      = config["model_params"]["filters1"]
    filters2      = config["model_params"]["filters2"]
    kernel_size1  = config["model_params"]["kernel_size1"]
    kernel_size2  = config["model_params"]["kernel_size2"]
    learning_rate = config["model_params"]["learning_rate"]
    momentum      = config["model_params"]["momentum"]
    optimizer_str = config["model_params"]["optimizer"]

    input_shape   = config["input_shape"]
    output_shape  = config["output_shape"]
    if optimizer_str.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_str.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_str}")

    model = Sequential([
        Conv1D(
            filters=filters1,
            kernel_size=kernel_size1,
            input_shape=(input_shape, 1)
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=filters2, kernel_size=kernel_size2),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Flatten(),


        Dense(100),
        LeakyReLU(alpha=0.01),

        Dense(40),
        LeakyReLU(alpha=0.01),

        Dense(10, activation="tanh"),

        Dense(output_shape, activation="linear")
    ])

    model.compile(optimizer=optimizer, loss="mse")
    return model
