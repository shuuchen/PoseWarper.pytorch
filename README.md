# PoseWarper.torch
A minimum implementation of pose warper https://arxiv.org/pdf/1906.04016.pdf

# How to use
- End-to-end style

Train the whole model all in one.
```python

```

- Baseline-warper separated style

Due to the big size of the model, it is difficult to train it as a whole on a commercial GPU with feasible batch sizes. Therefore, you can also separate the model to two models: baseline and warper, and train them sequentially. Actually, the author of original paper also trained in this way.
```python

```


# References
- [Learning Temporal Pose Estimation from Sparsely-Labeled Videos, 2019](https://arxiv.org/pdf/1906.04016.pdf)
- [Original implementation](https://github.com/facebookresearch/PoseWarper)
