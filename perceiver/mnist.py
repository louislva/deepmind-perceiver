import torch
import os


def load_mnist_model():
    return torch.load(
        os.path.join(
            os.path.dirname(__file__),
            '../models/mnist.pkl'
        )
    )
