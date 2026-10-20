import os

import torch
from setuptools import find_packages
from setuptools import setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
if torch_ver < [2, 6]:
    raise RuntimeError("Requires PyTorch >= 2.6")


def get_version():
    init_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "overdrive", "__init__.py")
    init_py = open(init_py_path).readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


setup(
    name="overdrive",
    version=get_version(),
    author="tkhe",
    url="https://github.com/tkhe/overdrive",
    description="overdrive is an open source toolbox for visual tasks.",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.10",
    install_requires=[
        "Pillow>=7.1",
        "cloudpickle",
        "hydra-core>=1.1",
        "imagesize",
        "matplotlib",
        "omegaconf>=2.1,<2.4",
        "opencv-python",
        "portalocker",
        "pycocotools>=2.0.2",
        "safetensors>=0.3.0",
        "scipy>1.5.1",
        "tabulate",
        "tensorboard",
        "termcolor>=1.1",
        "tqdm>4.29.0",
    ],
    extras_require={
        # optional dependencies, required by some features
        "all": [
            "cityscapesscripts",
            "fastapi",
            "lvis",
            "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
            "psutil",
            "pydantic",
            "pygments>=2.2",
            "shapely",
            "uvicorn[standard]",
        ],
    },
)
