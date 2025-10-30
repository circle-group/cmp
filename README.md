# CMP - Cubical Persistent Homology on CUDA

The PyTorch/CUDA implementation of Cubical Single-parameter and Multiparameter Persistent Homology for 2D images.
For details, see our paper CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations and [https://github.com/circle-group/cumperlay](https://github.com/circle-group/cumperlay).

## Installation

The library in this repo is tested with PyTorch 2.3, CUDA 12.2, and gcc-12/g++-12 and CUTLASS 3.5 (provided in [third_party/cutlass](third_party/cutlass/)).
Requirements:
- PyTorch 2.3+
- CUDA 12.2+
- gcc-12.2/g++-12.2 (or other versions supported by the CUDA version and this repository).
Ninja is recommended for faster builds.

To install:
```
pip install --no-build-isolation "git+https://github.com/circle-group/cmp.git#egg=cmp"
```

To test (further instructions/demos will be available soon):
```
python demo/demo.py
python demo/demo_npy.py
```

To install locally, git clone and then run:
```
pip install --no-build-isolation .
```

## Usage

Please see [https://github.com/circle-group/cumperlay](https://github.com/circle-group/cumperlay) for example usage.
(further instructions/demos will be available soon)

## Citation

If you use this codebase/library, or otherwise found our work valuable, please cite:
```
@InProceedings{Korkmaz_2025_CuMPerLay,
    author    = {Korkmaz, Caner and Nuwagira, Brighton and Coskunuzer, Baris and Birdal, Tolga},
    title     = {CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {27084-27094}
}
```