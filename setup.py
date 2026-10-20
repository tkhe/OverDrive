import glob
import os
import shutil

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
if torch_ver < [2, 6]:
    raise RuntimeError("Requires PyTorch >= 2.6")


def get_version():
    init_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "overdrive", "__init__.py")
    init_py = open(init_py_path).readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "overdrive", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))

    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(os.path.join(extensions_dir, "*.cu"))
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": ["-fvisibility=hidden"]}
    define_macros = []

    if (torch.cuda.is_available() and (CUDA_HOME is not None)) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "overdrive._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> list[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    overdrive/model_zoo.
    """
    # Use absolute paths while symlinking.
    source_configs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    destination = os.path.join(os.path.dirname(os.path.realpath(__file__)), "overdrive", "model_zoo", "configs")
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if os.path.exists(source_configs_dir):
        if os.path.islink(destination):
            os.unlink(destination)
        elif os.path.isdir(destination):
            shutil.rmtree(destination)

    if not os.path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.py", recursive=True)
    return config_paths


setup(
    name="overdrive",
    version=get_version(),
    author="tkhe",
    url="https://github.com/tkhe/overdrive",
    description="overdrive is an open source toolbox for visual tasks.",
    packages=find_packages(exclude=("configs", "tests*")),
    package_data={"overdrive.model_zoo": get_model_zoo_configs()},
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
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
