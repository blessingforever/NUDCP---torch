# NUDCP---torch
pytorch implementation of NUDCP [Paper](https://ieeexplore.ieee.org/abstract/document/8957276).

Primarily used for predicting of the medium transmission map in underwater environments.

# Source and Difference
The code is translated from the original python version from [Github repository](https://github.com/wangyanckxx/Enhancement-of-Underwater-Images-with-Statistical-Model-of-BL-and-Optimization-of-TM).

Tensor operations are changed from NumPy to Torch and can be applied in cuda.

# Citation
```bibtex
@article{Underwater Image Enhancement Method,
    author    = {Wei Song, Yan Wang, Dongmei Huang, Antonio Liotta, Cristian Perra},
    title     = {Enhancement of Underwater Images with Statistical Model of Background Light and Optimization of Transmission Map},
    journal   = {IEEE Transactions on Broadcasting},
    year      = {2019}
}
```
