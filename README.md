# DeepMind Perceiver (in PyTorch)

_Disclaimer: This is not official and I'm not affiliated with DeepMind._

My implementation of the _Perceiver: General Perception with Iterative Attention_. You can read more about the model [on DeepMind's website](https://deepmind.com/research/publications/Perceiver-General-Perception-with-Iterative-Attention).

I trained an MNIST model which you can find in `models/mnist.pkl` or by using `perceiver.load_mnist_model()`. It gets 96.02% on the test-data.

## Getting started

To run this you need PyTorch installed:

`pip3 install torch`

From `perceiver` you can import `Perceiver` or `PerceiverLogits`.

Then you can use it as such (or look in `examples.ipynb`):

```
from perceiver import Perceiver

model = Perceiver(
    input_channels, # <- How many channels in the input? E.g. 3 for RGB.
    input_shape, # <- How big is the input in the different dimensions? E.g. (28, 28) for MNIST
    fourier_bands=4, # <- How many bands should the positional encoding have?
    latents=64, # <- How many latent vectors?
    d_model=32, # <- Model dimensionality. Every pixel/token/latent vector will have this size.
    heads=8, # <- How many heads in self-attention? Cross-attention always has 1 head.
    latent_blocks=6, # <- How much latent self-attention for each cross attention with the input?
    dropout=0.1, # <- Dropout
    layers=8, # <- This will become two unique layer-blocks: layer 1 and layer 2-8 (using weight sharing).
)
```

The above model outputs the latents after the final layer. If you want logits instead, use the following model:

```
from perceiver import PerceiverLogits

model = PerceiverLogits(
    input_channels, # <- How many channels in the input? E.g. 3 for RGB.
    input_shape, # <- How big is the input in the different dimensions? E.g. (28, 28) for MNIST
    output_features, # <- How many different classes? E.g. 10 for MNIST.
    fourier_bands=4, # <- How many bands should the positional encoding have?
    latents=64, # <- How many latent vectors?
    d_model=32, # <- Model dimensionality. Every pixel/token/latent vector will have this size.
    heads=8, # <- How many heads in self-attention? Cross-attention always has 1 head.
    latent_blocks=6, # <- How much latent self-attention for each cross attention with the input?
    dropout=0.1, # <- Dropout
    layers=8, # <- This will become two unique layer-blocks: layer 1 and layer 2-8 (using weight sharing).
)
```

To use my pre-trained MNIST model (not very good):

```
from perceiver import load_mnist_model

model = load_mnist_model()
```

## TODO:

-   [x] Positional embedding generalized to _n_ dimensions (with fourier features)
-   [ ] Train other models (like CIFAR-100 or something not in the image domain)
-   [ ] Type indication
-   [ ] Unit tests for components of model
-   [ ] Package
