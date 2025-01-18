# PointNet++ Project Setup Guide

## Table of Contents

- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites

- CUDA 12.4
- Python 3.8 (recommended)
- Conda package manager

### Creating Virtual Environment

```bash
conda create -n pointnet_env python=3.8
conda activate pointnet_env
```

## Installation

### 1. PyTorch Installation

Install PyTorch with CUDA support:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 cudatoolkit=12.4 -c pytorch
```

### 2. PointNet++ Setup

Clone and install PointNet++:

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
```

### 3. Install pointnet2_ops_lib

Navigate to the ops library directory and install:

```bash
cd pointnet2_ops_lib
python setup.py install
```

#### Modified setup.py Configuration

```python
import glob
import os
import os.path as osp
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = [
    osp.join(_ext_src_root, "src", "sampling_gpu.cu"),
]
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

# Configure CUDA architecture
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_89,code=compute_89",
                ],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
```

### 4. Additional Dependencies

Install required Python packages:

```bash
pip install numpy sklearn tqdm torchnet plyfile open3d matplotlib
```

## Project Structure

```
your_project/
├── DATA/
│   ├── train/
│   └── test/
├── a_a_a.py
├── a_modelo_segmentation_v1.0.py
├── a_modelo_segmentation_v1.1.py
└── pointnet.py
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Compilation Errors**

   - Verify CUDA toolkit version matches PyTorch installation
   - Ensure CUDA paths are correctly set in environment variables

2. **CUDA Memory Errors**

   - Adjust batch size in configuration files
   - Monitor GPU memory usage during training

3. **CUDA Launch Blocking**
   - Add the following to your script:
     ```python
     import os
     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
     ```

### Additional Tips

- Keep GPU drivers up to date
- Monitor system resources during training
- Use `nvidia-smi` to check GPU usage and memory consumption

## Contributing

Please submit issues and pull requests for any improvements to the setup process.

## License

[Specify your license here]
