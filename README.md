1. Configuração do Ambiente Python

# Criar ambiente virtual (recomendado Python 3.7 ou 3.8)

conda create -n pointnet_env python=3.8
conda activate pointnet_env

# Instalar as dependências

## instalação do pytorch com a versão do cudsa 12.4

conda install pytorch==2.4.1 torchvision==0.19.1 cudatoolkit=12.4 -c pytorch

## Instalação do Pointnet++

### Clonar o repositório do Pointnet++

git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git

### Entrar no diretório

cd Pointnet2_PyTorch

### Instalar dependências

pip install -r requirements.txt

### Instalar o pointnet2_ops_lib

cd pointnet2_ops_lib
python setup.py install

modificar o setup.py
import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(**file**))
\_ext_src_root = osp.join("pointnet2_ops", "\_ext-src")
\_ext_sources = [
osp.join(_ext_src_root, "src", "sampling_gpu.cu"),
# Adicione outros arquivos .cu se necessário
]
\_ext_headers = glob.glob(osp.join(\_ext_src_root, "include", "\*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "\_version.py")).read())

# Set TORCH_CUDA_ARCH_LIST to include compute capability 8.9

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

setup(
name="pointnet2_ops",
version=**version**,
author="Erik Wijmans",
packages=find_packages(),
install_requires=requirements,
ext_modules=[
CUDAExtension(
name="pointnet2_ops.\_ext",
sources=\_ext_sources,
extra_compile_args={
"cxx": ["-O3"],
"nvcc": [
"-DCUDA_HAS_FP16=1",
"-D__CUDA_NO_HALF_OPERATORS__",
"-D__CUDA_NO_HALF_CONVERSIONS__",
"-D__CUDA_NO_HALF2_OPERATORS__",
# Include only architectures supported by your GPU
"-gencode=arch=compute_89,code=sm_89",
# Optionally include PTX code for future compatibility
"-gencode=arch=compute_89,code=compute_89",
],
},
include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
)
],
cmdclass={"build_ext": BuildExtension},
include_package_data=True,
)

# Outras Dependências

pip install numpy
pip install sklearn
pip install tqdm
pip install torchnet
pip install plyfile
pip install open3d
pip install matplotlib

# Estrutura do Projeto

seu_projeto/
├── DATA/
│ ├── train/
│ └── test/
├── a_a_a.py
├── a_modelo_segmentation_v1.0.py
├── a_modelo_segmentation_v1.1.py
└── pointnet.py

# Possíveis Problemas e Soluções

- Se houver erro ao compilar o Pointnet++, verifique se as versões do CUDA toolkit correspondem
- Em caso de erro de memória CUDA, ajuste o batch_size nos arquivos de configuração
- Para problemas com CUDA_LAUNCH_BLOCKING, adicione:

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
