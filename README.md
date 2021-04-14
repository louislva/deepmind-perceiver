# DeepMind Perceiver

_DISCLAIMER: This is not official and I'm not affiliated with DeepMind._

My implementation of DeepMind's Perceiver. You can read more about the model [on DeepMind's website](https://deepmind.com/research/publications/Perceiver-General-Perception-with-Iterative-Attention).

I trained an MNIST model which you can find in `models/mnist.pkl`. It get's 93.45% which is... so-so. In the bottom of this document are some to-do's that might help out:

## Getting started

To run this you need PyTorch installed:

`pip3 install torch`

From `perceiver` you can import `Perceiver` or `PerceiverLogits`.

Then you can use it as such (or look in `examples.ipynb`):

```
from perceiver import Perceiver

model = Perceiver(
    input_channels, # <- How many channels in the input? E.g. 3 for RGB.
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
    output_features, # <- How many different classes? E.g. 10 for MNIST.
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

-   [ ] Positional embedding generalized to _n_ dimensions
-   [ ] Type indication
-   [ ] Find a better MNIST model
-   [ ] Unit tests for components of model
-   [ ] Package
