import os
from surface_normal_uncertainty.download import download_file_from_google_drive

list_pip = [
    'torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html',
    'torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html',
    'torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
    'torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html',
    'torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html',
    'torch-cluster==1.5.8 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html',
    'torch-geometric==1.6.3',
    'Pillow',
    'numpy',
    'matplotlib',
    'argparse',
    'tqdm',
    'pytorch-lightning==1.1.2',
    'wandb',
    'opencv-python',
    'scikit-image==0.17.2',
    'pyrender',
    'trimesh',
    'kornia==0.4.1',
    'path',
    'pyhocon==0.3.57',
    'icecream==2.1.0',
    'scipy==1.7.0',
    'PyMCubes==0.1.2',
    'open3d==0.15.2',
    'wget']


for command in list_pip:
    print(command.strip())
    os.system(f'pip install {command.strip()}')

os.system('conda install openblas-devel -c anaconda -y')
os.system('pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"')
os.system('cd 3dvnet ; python -m pip install --editable . --user')

# download net weight
# https://drive.google.com/file/d/1lOgY9sbMRW73qNdJze9bPkM2cmfA8Re-/view?usp=share_link ---> surface_normal_uncertainty/checkpoints/scannet.pt
# https://drive.google.com/file/d/1CXgtwAXT3oBPgj6J1IRrA4wfmCD__6MF/view?usp=sharing ---> 3dvnet/3dvnet_weights/epoch=100-step=60700.ckpt
