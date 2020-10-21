# PoseWarper.torch
A minimum implementation of pose warper https://arxiv.org/pdf/1906.04016.pdf

This repository is for research use on the problem of keypoint estimation for videos with sparse labels.

# Environment
- This repository is tested successfully with PyTorch 1.4.0

# Dependencies
- This repository is dependent on the following repositories:
  - [HRNet](https://github.com/shuuchen/HRNet) as the baseline model.
  - [Deformable convolution](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) as the warper.

# How to use
- End-to-end style

Train the whole model all in one.
```python
from models.hrnet_warping import HRNet_Warping

model = HRNet_Warping(...)
```

- Baseline-warper separated style

Due to the big size of the model, it is difficult to train it as a whole on a commercial GPU with feasible batch sizes. Therefore, you can also separate the model to two models: baseline and warper, and train them sequentially. Actually, the author of the original paper also trained it in this way.
```python
from models.hrnet import HRNet
from models.warping import Warping

# create and train baseline model
baseline_model = HRNet(...)

...

# collect estimation results from baseline model, then fit waper model with the estimation results
warper_model = Warping(...)
```


# References
- [Learning Temporal Pose Estimation from Sparsely-Labeled Videos, 2019](https://arxiv.org/pdf/1906.04016.pdf)
- [Original implementation](https://github.com/facebookresearch/PoseWarper)
